import datetime
import logging
import math

import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from typing import Dict, List, Optional, Tuple


def get_category_unique_idxs(X: np.ndarray, categorical_names: Dict[int, List[str]]) -> \
        Tuple[Dict[int, Dict[float, float]], int]:
    """
    Creates a mapping that is used to translate the categories of each categorical variable in `X` to a unique index.

    Parameters
    ----------
    X
        Dataset whose categorical columns need to be remapped so that each category is assigned an unique index/
    categorical_names
        See `CategoricalEmbedding` constructor.

    Returns
    -------
    idx_map
        A mapping with the structure::

            {
            0: {0.0: 0.0, 1.0: 1.0}
            5: {0.0: 2.0, 2.0: 3:0}
            }
        The keys are the column indices for categorical variables whereas the values are mappings from the *unique
        values appearing in the column corresponding to said column index* to unique values. Each categorical variable
        is encoded in a range values equal to the `len` of the number of unique values appearing in its column.
    n_unique_levels
        The sum of the number of keys across the mappings in `idx_map`. This coincides with the sum of the lengths of
        the lists in `categorical_names` if all levels appear in `X`.

    Notes
    -----
        As shown in the example above, the categorical variable in column ``0``  can take 3 values (``0.0``, ``1.0``,
        ``2.0``) but only two (``0.0``, ``1.0``) are in `X[0,:]`. As a result, `2.0` does not appear as a key in the
        mapping for the first categorical variable and the unique index `2.0` is assigned to the level encoded as `0.0`
        of the categorical variable in the 5th column. This ensures the unique indices are contiguous.
    """

    cat_col_idxs = categorical_names.keys()
    idx_map = dict.fromkeys(cat_col_idxs)
    n_unique_levels = 0
    for idx in cat_col_idxs:
        current_encoding = np.unique(X[:, idx]).tolist()
        # might differ from the number of levels in `categorical_names` (e.g., some levels aren't in training data)
        actual_n_levels = len(current_encoding)
        idx_map[idx] = dict(
            zip(current_encoding, np.arange(n_unique_levels, n_unique_levels + actual_n_levels, dtype=np.float32))
        )
        n_unique_levels += actual_n_levels

    return idx_map, n_unique_levels


def remap_column(X: np.ndarray, col_idx: int, uniq_index_map: Dict[float, float]):
    """
    Remaps the categorical levels of a column in X to new values, specified by `categories_mapping`.

    Parameters
    ----------
    X
        Array whose categorical column will mapped to new levels.
    col_idx
        The index of the column to be remapped.
    uniq_index_map
        Map of unique values in `X[:, col_idx]` to new indices which are unique across all categorical variables.

    Notes
    -----
    This function changes `X` in place.
    """
    for original_category in np.unique(X[:, col_idx]):
        X[np.where(X[:, col_idx] == original_category), col_idx] = uniq_index_map[original_category]


def remap_categorical_levels(X: np.ndarray, uniq_idx_maps: Dict[int, Dict[float, float]]):
    """
    Remaps the categorical variables encoded in the columns of `X` such that each categorical level is assigned a unique
    integer. This is necessary in order to learn an embedding for each level of each categorical variable.

    Parameters
    ----------
    X
        Array whose categorical columns are to be remapped
    uniq_idx_maps
        Mapping containing a mapping of indices in `X` to new indices for each categorical variable. See
        `get_category_unique_idxs` for details
    """
    for cat_var_idx in uniq_idx_maps.keys():
        remap_column(X, cat_var_idx, uniq_idx_maps[cat_var_idx])


def permute_columns(X: np.ndarray, categorical_names: Dict[int, List[str]], inverse: bool = False) -> np.ndarray:
    """
    Permutes the columns of `X` such that the categorical variables occupy the first ``len(categorical_names.keys())``
    columns of the permuted `X` and the continuous variables occupy the remainder of the columns.

    Parameters
    ----------
    X
        Array to be permuted
    categorical_names
        See `alibi.utils.tensorflow.models.CategoricalEmbedding` constructor documentation.
    inverse
        If ``True`` the array is assumed to have been permuted through a previous application of `permute_columns`. The
        function will restore the order of the columns to the original one.
    """

    categorical_indices = list(categorical_names.keys())

    if inverse:
        new_array = np.zeros(X.shape)
        continuous_indices = [idx for idx in range(X.shape[1]) if idx not in categorical_indices]
        new_array[:, categorical_indices] = X[:, :len(categorical_indices)]
        new_array[:, continuous_indices] = X[:, len(categorical_indices):]

        return new_array

    return np.concatenate(
        [
            X[:, categorical_indices].reshape(X.shape[0], -1),
            np.delete(X, categorical_indices, axis=1).reshape(X.shape[0], -1)],
        axis=1
    )


class BatchGenerator:
    """
    A class which generates a dataset where each of the categorical variables in a given dataset `X` is set as a
    prediction target, in turn. For example, given the record ``[1, 2, 2.4, 2.6]`` where the first two elements
    represent encoded levels of two distinct categorical variables, the generated data will be
    ``[([1, 2.4, 2.6], [2]), ([2, 2.4, 2.6], [1])]``. This data is used to learn embeddings for the levels of a
    categorical variable inside the CategoricalEmbedding class.
    """

    def __init__(self, X: np.ndarray, n_categorical: int, batch_size: int, seed: Optional[int] = None):
        """
        Parameters
        ----------
        X
            Training data, with the following assumption:

                - The first `n_categorical` columns are categorical variables
                - `X` has been transformed such that the categories of the variables in the first `n_categorical`
                columns have been mapped to unique integers
        batch_size
            Number of training points returned in a batch
        seed
            Ensures the permutation of `X` is reproducible across different runs of the data loader.
        """

        self.seed = seed
        np.random.seed(seed)
        self.dataset = X
        self.dataset_permutation = np.random.permutation(np.arange(X.shape[0]))
        self.n_categorical = n_categorical
        self.n_datapoints = X.shape[0]
        self.n_features = X.shape[1]
        self.batch_size = batch_size
        # for each raw record, n_categories D-1 dimensional records are created. This may exceed batch size, so the
        # extra samples are stored in this buffer and returned in the next batch
        self.records_buffer = np.array([])
        self.targets_buffer = np.array([])
        self.n_processed = 0
        self.dataset_start_idx = 0

    def select_records(self, to_fill: int) -> int:
        """
        Calculates how many records from the original data can be augmented so that a batch of size `batch_size` is
        returned from the generator. It rounds up, since downstream functions will store the additional data point
        in self.buffer and return them in the next batch.

        Parameters
        ----------
        to_fill
            Number of rows empty in the current batch. This is not necessarily the batch size if `batch_size` is not
            divisible by `n_categories`, as the remainder number will be returned in the next batch.
        """
        # use ceil and roll over points we cannot fit to the next batch
        return math.ceil(to_fill / self.n_categorical)

    def create_targets(self, data_slice: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Creates targets for a number of records (a `data_slice`). For each record in slice, each categorical variable is
        a target in turn.

        Parameters
        ---------
        data_slice
            Records for which targets are to be created.

        Returns
        -------
        records, targets
            A tuple of data points and targets.
        """

        n_records, n_features = data_slice.shape[0], data_slice.shape[1]
        n_categorical = self.n_categorical
        n_new_records = n_records * n_categorical
        records, targets = np.zeros((n_new_records, n_features - 1)), np.zeros((n_new_records, 1))
        start_idx = 0
        # TODO: Could parallelize, but I doubt it we can speed it up b/c batch is small
        for idx in range(n_categorical):
            this_slice = slice(start_idx, start_idx + n_records)
            if n_categorical > 1:
                records[this_slice, :] = np.delete(data_slice, idx, axis=1)
            else:
                records[this_slice, :] = data_slice
            targets[this_slice] = data_slice[:, idx][:, np.newaxis]
            start_idx += n_records

        return records, targets

    def __call__(self, *args, **kwargs):

        while self.n_processed < self.n_datapoints:
            this_batch = np.zeros((self.batch_size, self.n_features - 1))
            this_batch_targets = np.zeros((self.batch_size, 1))

            # collect samples stored in the buffer (i.e., that we couldn't fit in the previous batch)
            n_buffered = self.records_buffer.shape[0]

            if n_buffered != 0:
                this_batch[:n_buffered, :] = self.records_buffer
                this_batch_targets[:n_buffered, :] = self.targets_buffer
                self.records_buffer = np.array([])
                self.targets_buffer = np.array([])

            # select and preprocess more data points
            batch_remaining_rows = self.batch_size - n_buffered
            n_selected = self.select_records(batch_remaining_rows)
            self.n_processed += n_selected
            records, targets = self.create_targets(
                self.dataset[self.dataset_permutation[self.dataset_start_idx:self.dataset_start_idx + n_selected], :]
            )
            self.dataset_start_idx += n_selected
            # store the samples we can't fit in a buffer an return in the next iteration
            if records.shape[0] > batch_remaining_rows:
                this_batch[n_buffered:, :] = records[:batch_remaining_rows, :]
                this_batch_targets[n_buffered:, :] = targets[:batch_remaining_rows, :]
                self.records_buffer = records[batch_remaining_rows:, :]
                self.targets_buffer = targets[batch_remaining_rows:, :]
            else:
                # k records left for batch of size b with k < b (last batch)
                if records.shape[0] != batch_remaining_rows:
                    this_batch = np.concatenate([this_batch[:n_buffered, :], records], axis=0)
                    this_batch_targets = np.concatenate([this_batch_targets[:n_buffered, :], targets], axis=0)
                else:
                    this_batch[n_buffered:, :] = records
                    this_batch_targets[n_buffered:, :] = targets

            yield this_batch, this_batch_targets

        # remaining data is returned
        self.n_processed = 0
        self.dataset_start_idx = 0
        if self.records_buffer.shape[0] != 0:
            last_batch, last_batch_target = self.records_buffer, self.targets_buffer
            self.records_buffer, self.targets_buffer = np.array([]), np.array([])

            yield last_batch, last_batch_target


class MixedDataEmbeddingLayer(tf.keras.layers.Layer):

    def __init__(self,
                 n_categorical: int,
                 categories: int,
                 embeddings_dim: int,
                 embeddings_mean: bool = False,
                 **kwargs):

        super(MixedDataEmbeddingLayer, self).__init__()
        self.n_categorical = n_categorical
        self.categories = categories
        self.embeddings_dim = embeddings_dim
        self.embeddings_mean = embeddings_mean
        self.embedding_layer = tf.keras.layers.Embedding(
            categories,
            embeddings_dim,
            input_length=n_categorical,
            **kwargs
        )

    def call(self, input, training=None):
        embedded_categorical = self.embedding_layer(input[:, :self.n_categorical])
        # TODO: check correct behaviour of averaging, inference
        if training:
            # combine embeddings
            embedding_part = tf.math.reduce_sum(embedded_categorical, axis=0, keepdims=True)
            if self.embeddings_mean:
                embedding_part *= (1/self.n_categorical)*embedding_part
            # add the continuous inputs
            return tf.concat([embedding_part, input[:, self.n_categorical:]], axis=1)
        # TODO: THIS MIGHT NEED A RESHAPE OP
        # concatenate embeddings in a single row
        return tf.concat(embedded_categorical, input[:, self.n_categorical])


DEFAULT_OPTIMIZER_PARAMS = {
    'learning_rate': 0.01,
    'momentum': 0.9,
}
"""
dict: Contains default parameters for the SGD optimizer that trains the embeddings.
"""


class CategoricalEmbedding:

    def __init__(self, categorical_names: Dict[int, List[str]], embedding_dim: int = 1, embeddings_mean: bool = False):
        """

        Parameters
        ----------
        categorical_names
            A dictionary with the following structure::

                {
                0: ['a', 'b', 'c'],
                5: ['a', 'b', 'c', 'd'],
                }
            The keys indicate that the 0th and 5th columns of `X` are categorical and the list contain the names of the
            categories. These are assumed to be ordinally encoded in `X` (e.g., the values in the 0th column of `X` are
            0., 1. and 2. whereas the values in the 5th column are 0., 1., 2., 3.).
        embeddings_mean
            If ``True``, the embeddings of a sequence of categories get combined by averaging, otherwise they
            get summed.
        """

        # TODO: categorical_names docstring can be used to improve all other algos who use it.

        self._fitted = False
        self.categorical_names = categorical_names
        self.n_categorical = len(self.categorical_names.keys())
        self.embedding_dim = embedding_dim
        self.embeddings_mean = embeddings_mean
        # maps the levels of each categorical to a new value so that no two levels across the categorical variables
        # have the same value
        self.index_map = {}  # type: Dict[int, Dict[float, float]]
        # number of levels to encode
        self.actual_embeddings = 0
        self.estimated_embeddings = sum([len(levels) for levels in categorical_names.values()])
        self.model = tf.keras.Sequential()
        self.category_embeddings = None  # set at fit time

    def fit(self,
            X: np.ndarray,
            epochs: int = 30,
            batch_size: int = 32,
            test_size: float = 0.2,
            optimizer: Optional['tf.keras.optimizers.Optimizer'] = None,
            optimizer_opts: Optional[Dict] = None,
            seed: Optional[int] = None,
            **kwargs,
            ):

        # TODO: CHECK THIS IS ENOUGH
        tf.random.set_seed(seed)
        # remap indices of the input data to unique indices & make sure categorical variables come first
        self.index_map, self.actual_embeddings = get_category_unique_idxs(X, self.categorical_names)
        if self.actual_embeddings != self.estimated_embeddings:
            logging.warning(
                f"{self.estimated_embeddings} levels were specified in categorical_names but only "
                f"{self.actual_embeddings} levels are in the training data. Some categories are missing from the "
                f"training data and will not be embedded."
            )
        remap_categorical_levels(X, self.index_map)
        X = permute_columns(X, self.categorical_names)

        X_train, X_val = train_test_split(X, test_size=test_size, random_state=seed)

        train_generator = BatchGenerator(X_train, n_categorical=self.n_categorical, batch_size=batch_size, seed=seed)
        train_dataset = tf.data.Dataset.from_generator(train_generator, (tf.float32, tf.float32))
        train_dataset = train_dataset.shuffle(self.n_categorical * X.shape[0], seed=seed, reshuffle_each_iteration=True)
        val_generator = BatchGenerator(X_val, n_categorical=self.n_categorical, batch_size=batch_size, seed=seed)
        val_dataset = tf.data.Dataset.from_generator(val_generator, (tf.float32, tf.float32))

        self.build_model()

        if optimizer is None:
            self._set_default_optimizer()
        else:
            if optimizer_opts is not None:
                self.optimizer = optimizer(**optimizer_opts)
            else:
                self.optimizer = optimizer()

        self.model.compile(
            optimizer=self.optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(), tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2)]
        )

        default_logdir = f"logs/scalars/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.model.fit(
            x=train_dataset,
            epochs=epochs,
            validation_data=val_dataset,
            verbose=kwargs.get('verbose', 1),
            callbacks=[tf.keras.callbacks.TensorBoard(log_dir=kwargs.get('log_dir', default_logdir))]
        )
        # TODO: NOT SURE THIS WILL WORK ...
        # self.category_embeddings = self.model.layers[1].embedding_layer
        self.category_embeddings = tf.keras.Model(inputs=self.model.inputs, outputs=self.model.layers[1].output)
        self._fitted = True

    def build_model(self):
        """
        Builds a model consisting of a n
        """
        # max(self.n_categorical - 1, 1) bc one of the categorical vars is a target
        # TODO: ADD INPUT LAYER
        self.model.add(MixedDataEmbeddingLayer(
            max(self.n_categorical - 1, 1),
            self.actual_embeddings,
            self.embedding_dim,
            embeddings_mean=self.embeddings_mean,
        ))
        self.model.add(tf.keras.layers.Dense(self.actual_embeddings))

    def _set_default_optimizer(self):
        """
        Initializes a default SGD optimizer for training the categorical embedding.
        """
        self.optimizer = tf.keras.optimizers.SGD(**DEFAULT_OPTIMIZER_PARAMS)

    def encode(self, X: np.ndarray) -> np.ndarray:
        """
        Encodes the mixed data in `X` using the embeddings for the categories.
        """
        remap_categorical_levels(X, self.index_map)
        X = permute_columns(X, self.categorical_names)
        return self.category_embeddings(X)


from alibi.datasets import fetch_adult

if __name__ == '__main__':
    adult = fetch_adult()
