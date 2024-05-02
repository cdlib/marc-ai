import argparse
import os
import tarfile
import warnings

import pytorch_lightning as lightning
import torch
import yaml
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from marcai.pl import MARCDataModule, SimilarityVectorModel
from marcai.utils import load_config


def train(name=None):
    config_path = "config.yaml"
    config = load_config(config_path)
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
        model_config["lr"],
        model_config["weight_decay"],
        model_config["optimizer"],
        model_config["batch_size"],
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

    # # Save config
    # config_filename = os.path.join(save_dir, "config.yaml")

    # with open(config_filename, "w") as f:
    #     dump = yaml.dump(config)
    #     f.write(dump)

    # # Compress model directory files
    # tar_path = f"{save_dir}/{name}.tar.gz"
    # with tarfile.open(tar_path, mode="w:gz") as archive:
    #     archive.add(save_dir, arcname=os.path.basename(save_dir))

    # Save model
    model.save_pretrained("marc-match-ai")


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--run-name", help="Name for training run", required=True)
    return parser


def main(args):
    train(args.run_name)


if __name__ == "__main__":
    args = args_parser().parse_args()
    main(args)
