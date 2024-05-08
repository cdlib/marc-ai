import argparse
import csv

from marcai.pl import SimilarityVectorModel
from marcai.predict import predict
from marcai.process import multiprocess_pairs
from marcai.utils import load_config
from marcai.utils.parsing import load_records, record_dict

import pandas as pd
from tqdm import tqdm

def args_parser():
    parser = argparse.ArgumentParser()
    required = parser.add_argument_group("required arguments")
    required.add_argument("-i", "--inputs", nargs="+", help="MARC files", required=True)
    required.add_argument(
        "-p",
        "--pair-indices",
        help="File containing indices of comparisons",
        required=True,
    )
    required.add_argument("-o", "--output", help="Output file", required=True)
    
    parser.add_argument(
        "-m",
        "--model",
        help="Path to the model directory, or HuggingFace model name",
        default="cdlib/marc-match-ai"
    )
    parser.add_argument("-C", "--chunksize", help="Chunk size", type=int, default=50000)
    parser.add_argument(
        "-P", "--processes", help="Number of processes for processing", type=int, default=1
    )
    parser.add_argument("-t", "--threshold", help="Threshold for matching", type=float)

    return parser


def main(args):

    # Load records
    print("Loading records...")
    records = []
    for path in args.inputs:
        records.extend([record_dict(r) for r in load_records(path)])

    records_df = pd.DataFrame(records)

    print(f"Loaded {len(records)} records.")

    model = SimilarityVectorModel.from_pretrained(args.model)

    print("Processing and comparing records...")
    written = False
    with open(args.pair_indices, "r") as indices_file:
        reader = csv.reader(indices_file)
        # Process records
        for df in tqdm(
            multiprocess_pairs(records_df, reader, args.chunksize, args.processes)
        ):
            prediction = predict(model, df[model.features])
            df.loc[:, "prediction"] = prediction.squeeze()

            if args.threshold:
                df = df[df["prediction"] >= args.threshold]

            if not df.empty:
                if not written:
                    df.to_csv(args.output, index=False)
                    written = True
                else:
                    df.to_csv(args.output, index=False, mode="a", header=False)


if __name__ == "__main__":
    args = args_parser().parse_args()
    main(args)
