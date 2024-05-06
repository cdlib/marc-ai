import argparse
import warnings

import pytorch_lightning as lightning
from pytorch_lightning.callbacks import EarlyStopping

from marcai.pl import MARCDataModule, SimilarityVectorModel
from marcai.utils import load_config


def train(name=None):
    config_path = "config.yaml"
    model_config = load_config(config_path)["model"]

    # Create data module from processed data
    warnings.filterwarnings("ignore", ".*does not have many workers.*")
    data = MARCDataModule(
        model_config["train_processed_path"],
        model_config["val_processed_path"],
        model_config["test_processed_path"],
        model_config["features"],
        model_config["batch_size"],
    )

    # Create model
    model = SimilarityVectorModel(
        lr=model_config["lr"],
        weight_decay=model_config["weight_decay"],
        optimizer=model_config["optimizer"],
        batch_size=model_config["batch_size"],
        features=model_config["features"]
    )

    callbacks = []
    if model_config["patience"] != -1:
        early_stop_callback = EarlyStopping(
            monitor="val_acc",
            min_delta=0.00,
            patience=model_config["patience"],
            verbose=False,
            mode="max",
        )
        callbacks.append(early_stop_callback)

    trainer = lightning.Trainer(
        max_epochs=model_config["max_epochs"], callbacks=callbacks, accelerator="cpu"
    )
    trainer.fit(model, data)

    # Save model
    model.save_pretrained(name)


def args_parser():
    parser = argparse.ArgumentParser()
    required = parser.add_argument_group("required arguments")
    required.add_argument("-n", "--run-name", help="Name for training run", required=True)
    return parser


def main(args):
    train(args.run_name)


if __name__ == "__main__":
    args = args_parser().parse_args()
    main(args)
