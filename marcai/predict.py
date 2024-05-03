import argparse

import numpy as np
import onnxruntime
import pandas as pd
import torch

from marcai.pl import SimilarityVectorModel
from marcai.utils import load_config


def sigmoid(x):
    return 1 / (1 + np.exp(-1 * x))


def predict_onnx(model_onnx_path, data):
    ort_session = onnxruntime.InferenceSession(model_onnx_path)

    x = data.to_numpy(dtype=np.float32)

    input_name = ort_session.get_inputs()[0].name
    ort_inputs = {input_name: x}
    ort_outs = np.array(ort_session.run(None, ort_inputs))
    ort_outs = sigmoid(ort_outs)

    return ort_outs


def predict(model, data):
    x = torch.stack(
        [torch.tensor(data.values[i].astype(float)) for i in range(len(data))], dim=0
    ).float()

    model.eval()
    with torch.no_grad():
        prediction = model.predict(x).detach().numpy()

    return prediction


def args_parser():
    parser = argparse.ArgumentParser()

    required = parser.add_argument_group("required arguments")
    required.add_argument(
        "-i", "--input", help="Path to preprocessed data file", required=True
    )
    required.add_argument("-o", "--output", help="Output path", required=True)
    parser.add_argument(
        "-m",
        "--model",
        help="Path to the model directory, or HuggingFace model name",
        default="cdlib/marc-match-ai",
    )
    parser.add_argument(
        "--chunksize",
        help="Chunk size for reading and predicting",
        default=1024,
        type=int,
    )
    parser.add_argument(
        "--config",
        help="Path to the config.yaml file",
        default="config.yaml",
    )
    return parser


def main(args):
    # Load config.yaml
    config = load_config(args.config)

    # Load data
    data = pd.read_csv(args.input, chunksize=args.chunksize)

    model = SimilarityVectorModel.from_pretrained(args.model)

    written = False
    for chunk in data:
        prediction = predict(model, chunk[config["model"]["features"]])

        # Add prediction to chunk
        chunk["prediction"] = prediction.squeeze()

        # Append to CSV
        if not written:
            chunk.to_csv(args.output, index=False)
            written = True
        else:
            chunk.to_csv(args.output, mode="a", header=False, index=False)


if __name__ == "__main__":
    args = args_parser().parse_args()
    main(args)
