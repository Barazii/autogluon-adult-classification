import json
import numpy as np

try:
    from sagemaker_containers.beta.framework import encoders, worker
except ImportError:
    worker = None


def model_fn(model_dir):
    pass


def input_fn(probabilities, xyz):
    if probabilities == None or len(probabilities) == 0:
        raise ValueError("No predictions/probabilities received from the model.")
    probabilities = json.loads(probabilities)["probabilities"]
    predictions = np.argmax(np.array(probabilities), axis=-1)
    return predictions


def predict_fn(predictions, xyz):
    return predictions


def output_fn(predictions, accept):
    return (
        worker.Response(encoders.encode(predictions, accept), mimetype=accept)
        if worker
        else (predictions, accept)
    )
