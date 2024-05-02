import argparse

import numpy as np
import onnxruntime
import pandas as pd

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

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", help="Path to preprocessed data file", required=True
    )
    parser.add_argument("-o", "--output", help="Output path", required=True)
    parser.add_argument(
        "-m",
        "--model-dir",
        help="Directory containing model ONNX and YAML files",
        required=True,
    )
    parser.add_argument(
        "--chunksize",
        help="Chunk size for reading and predicting",
        default=1024,
        type=int,
    )
    return parser


def main(args):
    config_path = f"{args.model_dir}/config.yaml"
    model_onnx = f"{args.model_dir}/model.onnx"

    config = load_config(config_path)

    # Load data
    data = pd.read_csv(args.input, chunksize=args.chunksize)

    written = False
    for chunk in data:
        # Limit columns to model input features
        input_df = chunk[config["model"]["features"]]

        prediction = predict_onnx(model_onnx, input_df)

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
