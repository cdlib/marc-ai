# MARC Record Matching with Bibliographic Metadata

Check out our [pre-trained model](https://huggingface.co/cdlib/marc-match-ai) and [interactive demo](https://huggingface.co/spaces/cdlib/marc-match-ai-demo) on HuggingFace!

Traditional matching of MARC (Machine-Readable Cataloging) records has relied heavily on identifiers like OCLC numbers, ISBNs, LCCNs, etc. assigned by catalogers. However, this approach struggles with records having incorrect identifiers or lacking them altogether. This model has been developed to match MARC records based solely on their bibliographic metadata (title, author, publisher, etc.), enabling successful matches even when identifiers are missing or inaccurate.

We have primarily focused on MARC records for English monographs contributed to the HathiTrust by partnering institutions. The future direction of this repository is uncertain, but we plan to develop a new dataset and model encompassing a broader range of languages and publication locations. If you would like to contribute datasets, results, or models/methods, please contact us. We are eager to connect with others working on MARC record matching.

## Key Features
- Bibliographic Metadata Matching: Performs matching based solely on bibliographic data, eliminating the need for identifiers.
- Text Field Flexibility: Accommodates minor variations in bibliographic metadata fields for accurate matching.
- Adjustable Matching Threshold: Allows tuning the balance between false positives and false negatives based on specific use cases.

## Installation
Installing the `marcai` package is as simple as installing this Git repository as a Python package:
```
pip install git+https://github.com/cdlib/marc-ai.git
```

## Usage
The `marcai` package comes with a command-line interface offering a suite of commands for processing data, training models, and making predictions. All commands have their own help functions, which can be accessed by running `marc-ai <command> --help`.

### Processing data
`marc-ai process` takes a file containing MARC records and a CSV containing indices of record comparisons, and calculates similarity scores for several fields in the MARC records. These similarity values serve as the input features to the machine learning model.
```
usage: marc-ai process [-h] -i INPUTS [INPUTS ...] -o OUTPUT [-p PAIR_INDICES] [-C CHUNKSIZE] [-P PROCESSES]

options:
  -h, --help            show this help message and exit
  -C CHUNKSIZE, --chunksize CHUNKSIZE
                        Number of comparisons per job
  -P PROCESSES, --processes PROCESSES
                        Number of processes to run in parallel.

required arguments:
  -i INPUTS [INPUTS ...], --inputs INPUTS [INPUTS ...]
                        MARC files
  -o OUTPUT, --output OUTPUT
                        Output file
  -p PAIR_INDICES, --pair-indices PAIR_INDICES
                        File containing comma separated indices of comparisons (one comparison per line)
```

### Training a model
`marc-ai train` trains a model with the hyperparameters defined in `config.yaml`, including the paths to __processed__ dataset splits.
```
usage: marc-ai train [-h] -n RUN_NAME

options:
  -h, --help            show this help message and exit

required arguments:
  -n RUN_NAME, --run-name RUN_NAME
                        Name for training runmarc-ai train --name <training_run_name>
```
A directory for the training run will be created with the model and hyperparameters.

### Making predictions
`marc-ai predict` takes the output from `marc-ai process` and a trained model, and runs the similarity scores through the model to produce match confidence scores. By default it will use our HuggingFace pretrained model, `cdlib/marc-match-ai`.

```
usage: marc-ai predict [-h] -i INPUT -o OUTPUT [-m MODEL] [--chunksize CHUNKSIZE] [--config CONFIG]

options:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        Path to the model directory, or HuggingFace model name
  --chunksize CHUNKSIZE
                        Chunk size for reading and predicting
  --config CONFIG       Path to the config.yaml file

required arguments:
  -i INPUT, --input INPUT
                        Path to preprocessed data file
  -o OUTPUT, --output OUTPUT
                        Output path
```

### Finding matches without I/O
`marc-ai find_matches` combines the commands for processing and predicting to cut out the unnecessary step of saving similarity values to disk. This is substantially faster when working with large amounts of data.
```
usage: marc-ai find_matches [-h] -i INPUTS [INPUTS ...] -p PAIR_INDICES -m MODEL_DIR -o OUTPUT [-C CHUNKSIZE]
                            [-P PROCESSES] [-t THRESHOLD]

options:
  -h, --help            show this help message and exit
  -C CHUNKSIZE, --chunksize CHUNKSIZE
                        Chunk size
  -P PROCESSES, --processes PROCESSES
                        Number of processes
  -t THRESHOLD, --threshold THRESHOLD
                        Threshold for matching

required arguments:
  -i INPUTS [INPUTS ...], --inputs INPUTS [INPUTS ...]
                        MARC files
  -p PAIR_INDICES, --pair-indices PAIR_INDICES
                        File containing indices of comparisons
  -m MODEL_DIR, --model-dir MODEL_DIR
                        Directory containing model ONNX and YAML files
  -o OUTPUT, --output OUTPUT
                        Output file
```

## Analysis (Jupyter Notebooks)

We have provided Jupyter notebooks containing analyses conducted during this project. The purpose of these notebooks is to examine the dataset and our current model results, as well as to share the methodologies employed throughout the project.

## Dataset

The initial dataset originates from HathiTrust contributors; however, the records have been anonymized, with identifiers and custom fields removed. This dataset was specifically designed to create and evaluate record pairing methods based on content alone, making it unsuitable for pairing records using both content and identifiers or evaluating the entire HathiTrust collection. The HathiTrust data is licensed under CC0, with certain caveats detailed in the [LICENSE.md](datasets/ht/LICENSE.md) file.

The data is real and may contain some errors and peculiarities due to the way HathiTrust combines monograph records. We plan to collaborate with HathiTrust to make this dataset more accessible to a wider audience, perhaps on Hugging Face. We welcome feedback on the format or any issues to improve its usefulness.

## Results

The results folder contains our model's outcomes, as well as some basic attempts at string matching and fuzzy string matching. These findings are used by the analysis notebooks to compare and contrast the performance of various methods.



