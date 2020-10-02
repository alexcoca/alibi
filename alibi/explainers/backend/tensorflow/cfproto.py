import numpy as np
import tensorflow as tf

from typing import Any, Dict, Mapping, Optional, Union

if TYPE_CHECKING:  # pragma: no cover
    import keras


PROTO_METHOD_OPTS = {
    'hyperparameters': {
        'theta': 0.,
    },
    'seach_opts': {},
}
# TODO: DOCUMENT
# TODO: MOVE TO IMPLEMENTATION CLASS
# TODO: TYPE IT


def proto_ae(counterfactual: tf.Variable, target_proto: tf.Tensor,  enc_model: Optional[tf.keras.Model, keras.Model]):
    return tf.square(tf.norm(enc_model(counterfactual) - target_proto, ord=2))
# TODO: DOCS


def proto_kdtree(counterfactual: tf.Variable, target_proto: tf.Tensor):
    return tf.square(tf.norm(counterfactual - target_proto, ord=2))
# TODO: DOCS


PROTO_LOSS_SPEC_WHITEBOX = {
    'proto': {'fcn': proto_ae, 'kwargs': {}}
}  # type: Dict[str, Mapping[str, Any]]
# TODO: DOCUMENT (inc various options for what `proto` loss can be


class TFProtoCounterfactualOptimizer:

    def __init__(self,
                 predictor,
                 ae_model: Union[tf.keras.Model, 'keras.Model', None],
                 enc_model: Union[tf.keras.Model, 'keras.Model', None]):

        self.ae_model = ae_model
        self.enc_model = enc_model

        # initialised by caller at explain time through compute_proto call
        self.proto = None

    def compute_proto(self, X: np.ndarray):

        self.proto = None


