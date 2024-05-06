import argparse
import concurrent.futures
import csv
import time
from multiprocessing import get_context

import numpy as np
import pandas as pd
from more_itertools import chunked

import marcai.processing.comparisons as comps
import marcai.processing.normalizations as norms
from marcai.utils.parsing import load_records, record_dict


def multiprocess_pairs(
    records_df,
    pair_indices,
    chunksize=50000,
    processes=1,
):
    # Create chunked iterator
    pairs_chunked = chunked(pair_indices, chunksize)

    # Create processing jobs
    max_jobs = processes * 2

    context = get_context("fork")

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=processes, mp_context=context
    ) as executor:
        futures = set()
        done = set()
        first_spawn = True

        while futures or first_spawn:
            if first_spawn:
                spawn_count = max_jobs
                first_spawn = False
            else:
                # Wait for a job to complete
                done, futures = concurrent.futures.wait(
                    futures, return_when=concurrent.futures.FIRST_COMPLETED
                )
                spawn_count = max_jobs - len(futures)

                for future in done:
                    # Get job's output
                    df = future.result()

                    # Yield output
                    yield df

            # Spawn jobs
            for _ in range(spawn_count):
                pairs_chunk = next(pairs_chunked, None)

                if pairs_chunk is None:
                    break

                indices = np.array(pairs_chunk).astype(int)

                left_indices = indices[:, 0]
                right_indices = indices[:, 1]

                left_records = records_df.iloc[left_indices].reset_index(drop=True)
                right_records = records_df.iloc[right_indices].reset_index(drop=True)

                futures.add(executor.submit(process, left_records, right_records))


def process(df0, df1):
    normalize_fields = [
        "author_names",
        "corporate_names",
        "meeting_names",
        "publisher",
        "title",
        "title_a",
        "title_b",
        "title_c",
        "title_p",
    ]

    # Normalize text fields
    for field in normalize_fields:
        df0[field] = norms.lowercase(df0[field])
        df1[field] = norms.lowercase(df1[field])

        df0[field] = norms.remove_punctuation(df0[field])
        df1[field] = norms.remove_punctuation(df1[field])

        df0[field] = norms.remove_diacritics(df0[field])
        df1[field] = norms.remove_diacritics(df1[field])

        df0[field] = norms.normalize_whitespace(df0[field])
        df1[field] = norms.normalize_whitespace(df1[field])

    # Compare fields
    result_df = pd.DataFrame()

    result_df["id_0"] = df0["id"]
    result_df["id_1"] = df1["id"]

    # Token sort ratio
    result_df["publisher"] = comps.token_sort_similarity(
        df0["publisher"], df1["publisher"], null_value=0.5
    )

    author_names = comps.token_sort_similarity(
        df0["author_names"], df1["author_names"], null_value=np.nan
    )
    corporate_names = comps.token_sort_similarity(
        df0["corporate_names"], df1["corporate_names"], null_value=np.nan
    )
    meeting_names = comps.token_sort_similarity(
        df0["meeting_names"], df1["meeting_names"], null_value=np.nan
    )
    authors = pd.concat([author_names, corporate_names, meeting_names], axis=1)

    # Take max of author comparisons
    result_df["author"] = comps.maximum(authors, null_value=0.5)

    # Weighted title comparison
    weights = {"title_a": 1, "raw": 0, "title_p": 1}

    result_df["title_agg"] = comps.column_aggregate_similarity(
        df0[weights.keys()], df1[weights.keys()], weights.values(), null_value=0
    )

    # Token set similarity
    result_df["title_tokenset"] = comps.token_set_similarity(
        df0["title"], df1["title"], null_value=0
    )

    # Pagination
    result_df["pagination"] = comps.pagination_match(
        df0["pagination"], df1["pagination"], null_value=0.5
    )

    # Dates
    result_df["pub_date"] = comps.year_similarity(
        df0["pub_date"], df1["pub_date"], null_value=0.5, exp_coeff=0.15
    )

    # Pub place
    result_df["pub_place"] = comps.equal(
        df0["pub_place"], df1["pub_place"], null_value=0.5
    )

    # CID/Label
    result_df["cid"] = comps.equal(df0["cid"], df1["cid"], null_value=0.5)

    return result_df


def args_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    required = parser.add_argument_group("required arguments")
    required.add_argument("-i", "--inputs", nargs="+", help="MARC files", required=True)
    required.add_argument("-o", "--output", help="Output file", required=True)
    required.add_argument(
        "-p",
        "--pair-indices",
        help="File containing comma separated indices of comparisons (one comparison per line)",
    )

    parser.add_argument(
        "-C",
        "--chunksize",
        type=int,
        help="Number of comparisons per job",
        default=50000,
    )

    parser.add_argument(
        "-P",
        "--processes",
        type=int,
        help="Number of processes to run in parallel.",
        default=1,
    )

    return parser


def main(args):
    start = time.time()

    # Load records
    print("Loading records...")
    records = []
    for path in args.inputs:
        records.extend([record_dict(r) for r in load_records(path)])

    records_df = pd.DataFrame(records)

    print(f"Loaded {len(records)} records.")

    print("Processing records...")
    # Process records
    written = False
    with open(args.pair_indices, "r") as indices_file:
        reader = csv.reader(indices_file)

        for df in multiprocess_pairs(
            records_df, reader, args.chunksize, args.processes
        ):
            if not written:
                # Write header
                df.to_csv(args.output, mode="w", header=True, index=False)
                written = True
            else:
                # Write rows of df to output CSV
                df.to_csv(args.output, mode="a", header=False, index=False)

    end = time.time()
    print(f"Processed {len(records)} records.")
    print(f"Time elapsed: {end - start:.2f} seconds.")


if __name__ == "__main__":
    args = args_parser().parse_args()
    main(args)
