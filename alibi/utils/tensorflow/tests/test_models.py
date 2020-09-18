# flake8: noqa E731
import pytest
import random

import numpy as np
import tensorflow as tf

from alibi.utils.tensorflow.models import BatchGenerator, permute_columns


# TODO: This is a general fixture but not sure how to make it available to this file (except by adding a conftest.py
#  here but that does not make it available outside this directory ...)

@pytest.fixture
def tabular_data_generator(request):
    """
    Generates a random tabular dataset with `n_records'. The first `n_categorical` columns are categorical variables
    ith values chosen randomly according to `category_levels`. The following `n_continuous` columns are randomly
    sampled from the uniform distribution.

    `category_levels` should be a list of len `n_categorical` where the each element is the number of levels for the
    categorical column with the same index (e.g., first element is the number of levels for the var in the first col.).
    """

    n_records, n_categorical, n_continuous, category_levels, permuted, seed = request.param
    np.random.seed(seed[0])
    random.seed(seed[0])
    b, a = 5, 2  # continuous data range, generate using (b-a)*U[0,1) + a

    if len(category_levels) != n_categorical:
        raise ValueError(
            f"Please specify the number of categorical level for each variable. "
            f"Specified {n_categorical} categorical variables and {len(category_levels)} levels.")

    n_features = n_categorical + n_continuous
    X = np.zeros((n_records, n_features))

    # assign col indices to a data type depending on whether the cols should be permuted or not
    if permuted:
        categorical_indices = random.sample(range(n_features), n_categorical)
    else:
        categorical_indices = list(range(n_categorical))
    continuous_indices = [idx for idx in range(n_features) if idx not in categorical_indices]

    # generate data
    for col_idx, n_levels in zip(categorical_indices, category_levels):
        X[:, col_idx] = np.random.randint(0, n_levels, size=(n_records,))
    X[:, continuous_indices] = np.random.random_sample(size=(n_records, n_continuous))*(b - a) + a

    # create category map
    category_map_values = []
    for col_idx, n_categories in zip(categorical_indices, category_levels):
        this_var_categories = [f'feature_{col_idx}_name_{i}' for i in range(n_categories)]
        category_map_values.append(this_var_categories)
    category_map = dict(zip(categorical_indices, category_map_values))

    return {'data': X, 'category_map': category_map}


# TODO: CAN IMPROVE TEST CASE GENERATION
seed = [None, ]  # pytest somehow throws an error if you try without the list
# data generation params
n_records, n_categorical, n_continuous = 50, 12, 3
category_levels = [4, 12, 3, 6, 4, 4, 12, 3, 6, 4, 6, 8]
permuted = False  # the categorical variables occupy the first `n_categorical` columns

data_gen_params = [(n_records, n_categorical, n_continuous, category_levels, permuted, seed), ]

# test params
batch_sizes = [9, 10, 32, 64, 63, 50, 49]


def ids_generator(params):
    fmt = 'n_records_{}_n_categorical_{}_n_continuous_{}_category_levels_{}_permuted_{}_seed_{}'
    return fmt.format(*params)


@pytest.mark.parametrize('tabular_data_generator', data_gen_params, indirect=True, ids=ids_generator)
@pytest.mark.parametrize('batch_size', batch_sizes, ids='batch_size={}'.format)
@pytest.mark.parametrize('seed', seed, ids='seed={}'.format)
def test_BatchGenerator(tabular_data_generator, batch_size, seed):
    """
    Tests data generator for categorical embeddings training. Checks if:

        - the concatenation of all returned data has expected shape
        - the data points and target batches have equal leading dimension
        - the correctness of the data returned (both categorical and continuous)
        - the batches returned have correct size
        - the generator has erroneously returned rows of zero
    """

    # absolute tol for 0.0 equality comparisons
    atol = 1e-6

    X = tabular_data_generator['data']
    n_categoricals = len(tabular_data_generator['category_map'].keys())

    # create dataset
    generator = BatchGenerator(X, n_categorical=n_categoricals, batch_size=batch_size, seed=seed)
    dataset = tf.data.Dataset.from_generator(generator, (tf.float32, tf.float32))
    # collect data from the generator
    returned_x, returned_y = [], []
    for x, y in dataset:
        returned_x.append(x)
        returned_y.append(y)
        assert x.shape[0] == y.shape[0]
    for idx in range(len(returned_x) - 1):  # last batch can be a different size
        assert returned_x[idx].shape[0] == returned_y[idx].shape[0]
    returned_records = tf.concat(returned_x, axis=0).numpy()
    returned_targets = tf.concat(returned_y, axis=0).numpy()

    # find the index of the returned records in the original dataset
    raw_data_indices = np.zeros((returned_records.shape[0],))
    for row in range(returned_records.shape[0]):
        differences = X[:, n_categoricals:] - returned_records[row, n_categoricals - 1:]
        orig_idx = np.where(np.isclose(differences.sum(axis=1), 0.0, atol=atol) == 1)
        # test fails here sometimes, I think it is because the returned records matches two rows of X.
        raw_data_indices[row] = (orig_idx[0].item())
    # check data returned is correct
    for uniq_idx in np.unique(raw_data_indices):
        # check that each data point appears n_categories times in the result
        locs = raw_data_indices == uniq_idx
        assert locs.sum() == n_categoricals
        # continuous data is correct

        np.testing.assert_allclose(
            returned_records[locs, n_categoricals - 1:],
            np.repeat(X[int(uniq_idx), n_categoricals:][np.newaxis, :], sum(locs), axis=0),
            atol=atol
        )

        # because of shuffling (both by tensorflow and BatchGenerator internal), the order of the returned targets
        # differs from the order of the X[int(uniq_idx), :n_categoricals] vector but they should have the same sum
        assert np.isclose(
            returned_targets[locs].sum(), X[int(uniq_idx), :n_categoricals].reshape(-1, 1).sum(), atol=atol,
        )
        # note: in theory, one could test if all possible d-1 records are returned by excluding in turn each number
        # from X[int(uniq_idx), :n_categoricals] and checking if it's in returned_records[locs]. But there is an issue
        # with this check: if we have e.g, [2, 2, 4, ...] then we check for [2, 4, ...] twice - it's a bit tricky to
        # deal with this multiplicity (and pointless) but in theory each subset should be found in returned_records at
        # least once and that can be tested

    assert returned_records.shape == (n_categoricals * X.shape[0], X.shape[1] - 1)
    assert returned_targets.shape == (n_categoricals * X.shape[0], 1)
    # batch is initialised with rows of zeros, check that actual data only is returned
    assert not np.any(np.isclose(returned_records.sum(axis=1), 0.0, atol=atol))


permuted = True  # generate the columns in the permuted order to test function that reorders them
data_gen_params = [(n_records, n_categorical, n_continuous, category_levels, permuted, seed), ]


@pytest.mark.parametrize('tabular_data_generator', data_gen_params, indirect=True, ids=ids_generator)
def test_permute_columns(tabular_data_generator):
    """
    Tests permutation function for tabular data. In particular:

        - Tests that the categorical columns are first after its application with argument `inverse=False`
        - Tests that the original array is recovered if the function is applied with argument `inverse=True` to an array
        obtained by applying the function with `inverse=False` (aka that we can invert the permutation)
    """
    atol = 1e-5  # absolute tolerance for floating point equality comparisons

    X = tabular_data_generator['data']
    categorical_names = tabular_data_generator['category_map']
    categorical_subarray = X[:, list(categorical_names.keys())]

    X_permuted = permute_columns(X, categorical_names)
    orig_X = permute_columns(X_permuted, categorical_names, inverse=True)

    np.testing.assert_allclose(X_permuted[:, :len(categorical_names)], categorical_subarray, atol=atol)
    np.testing.assert_allclose(X, orig_X, atol=atol)
