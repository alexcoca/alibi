import math

import numpy as np

from typing import Dict, List, Optional, Tuple


class MixedDataEncoder:

    def __init__(self, categorical_names: Optional[Dict[int, List[str]]] = None):
        pass

    def __call__(self, *args, **kwargs):
        pass


class CategoricalEmbedding:

    def __init__(self,
                 n_features: int,
                 embedding_dim: int = 1,
                 categorical_names: Optional[Dict[int, List[str]]] = None):

        n_embeddings = len(categorical_names.keys())
        self.embeddings = None

    def fit(self, X: np.ndarray):
        pass


class BatchGenerator:
    """
    A class which generates a dataset where each of the categorical variables in a given dataset `X` is set as a
    prediction target, in turn. For example, given the record ``[1, 2, 2.4, 2.6]`` where the first two elements
    represent encoded levels of two distinct categorical variables, the generated data will be
    ``[([1, 2.4, 2.6], [2]), ([2, 2.4, 2.6], [1])]``. This data is used to learn embeddings for the levels of a
    categorical variable inside #TODO: ADD OBJECT NAME.
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
        # TODO: FINISH DOCSTRING

        """

        n_records, n_features = data_slice.shape[0], data_slice.shape[1]
        n_categorical = self.n_categorical
        n_new_records = n_records * n_categorical
        records, targets = np.zeros((n_new_records, n_features - 1)), np.zeros((n_new_records, 1))
        start_idx = 0
        # TODO: Could parallelize, but I doubt it we can speed it up b/c batch is small
        for idx in range(n_categorical):
            this_slice = slice(start_idx, start_idx + n_records)
            records[this_slice, :] = np.delete(data_slice, idx, axis=1)
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

        # any remaining data is returned
        self.n_processed = 0
        self.dataset_start_idx = 0
        if self.records_buffer.shape[0] != 0:
            last_batch, last_batch_target = self.records_buffer, self.targets_buffer
            self.records_buffer, self.targets_buffer = np.array([]), np.array([])

            yield last_batch, last_batch_target
