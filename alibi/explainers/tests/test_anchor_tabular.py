# flake8: noqa E731

from alibi.explainers import AnchorTabular, AnchorBaseBeam, AnchorExplanation
import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier


@pytest.mark.parametrize('predict_type', ('proba', 'class'))
@pytest.mark.parametrize('threshold', (0.9, 0.95))
def test_iris(predict_type, threshold):
    # load iris dataset
    dataset = load_iris()
    feature_names = dataset.feature_names

    # define train and test set
    idx = 145
    X_train, Y_train = dataset.data[:idx, :], dataset.target[:idx]
    X_test, Y_test = dataset.data[idx + 1:, :], dataset.target[idx + 1:]  # noqa F841

    # fit random forest to training data
    np.random.seed(0)
    clf = RandomForestClassifier(n_estimators=50)
    clf.fit(X_train, Y_train)

    # define prediction function
    if predict_type == 'proba':
        predict_fn = lambda x: clf.predict_proba(x)
    elif predict_type == 'class':
        predict_fn = lambda x: clf.predict(x)

    # test explainer initialization
    explainer = AnchorTabular(predict_fn, feature_names)
    assert explainer.predict_fn(X_test[0].reshape(1, -1)).shape == (1,)

    # test explainer fit: shape and binning of ordinal features
    explainer.fit(X_train, disc_perc=(25, 50, 75))
    assert explainer.train_data.shape == explainer.d_train_data.shape == (145, 4)
    assert (np.unique(explainer.d_train_data) == np.array([0., 1., 2., 3.])).all()
    assert not explainer.categorical_features

    # test sampling function
    if predict_type == 'proba':
        explainer.instance_label = np.argmax(predict_fn(X_test[0, :].reshape(1, -1)), axis=1)
    else:
        explainer.instance_label = predict_fn(X_test[0, :].reshape(1, -1))[0]
    explainer.build_sampling_lookups(X_test[0, :])
    anchor = list(explainer.enc2feat_idx.keys())
    nb_samples = 5
    raw_data, data, labels, _ = explainer.sampler(anchor, nb_samples)
    assert len(explainer.enc2feat_idx) == data.shape[1]

    # test mapping dictionary used for sampling
    assert (set(explainer.ord_lookup.keys() | set(explainer.cat_lookup.keys()))) == set(explainer.enc2feat_idx.keys())

    # test explanation
    explain_defaults = {'delta': 0.1,
                        'epsilon': 0.15,
                        'batch_size': 100,
                        'desired_confidence': threshold,
                        'max_anchor_size': None,
                        }
    anchor = AnchorBaseBeam.anchor_beam(explainer.sampler,
                                        **explain_defaults,
                                        )
    explainer.add_names_to_exp(anchor)
    exp = AnchorExplanation('tabular', anchor)
    assert exp.precision() >= threshold
    assert exp.coverage() >= 0.05
