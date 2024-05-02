import argparse
from . import train, predict, process, find_matches


def main():
    parser = argparse.ArgumentParser(
        description="Command-line interface for marcai package"
    )
    subparsers = parser.add_subparsers(required=True)

    train_parser = subparsers.add_parser(
        "train", parents=[train.args_parser()], help="Train a model", add_help=False
    )
    predict_parser = subparsers.add_parser(
        "predict",
        parents=[predict.args_parser()],
        help="Make predictions using a trained model",
        add_help=False,
    )
    process_parser = subparsers.add_parser(
        "process", parents=[process.args_parser()], help="Process data", add_help=False
    )
    find_matches_parser = subparsers.add_parser(
        "find_matches",
        parents=[find_matches.args_parser()],
        help="Find matches in data",
        add_help=False,
    )

    train_parser.set_defaults(func=train.main)
    predict_parser.set_defaults(func=predict.main)
    process_parser.set_defaults(func=process.main)
    find_matches_parser.set_defaults(func=find_matches.main)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
