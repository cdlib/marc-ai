# MARC Record Matching with Bibliographic Metadata

Check out our [pre-trained model](https://huggingface.co/cdlib/marc-match-ai) and the [interactive demo](https://huggingface.co/spaces/cdlib/marc-match-ai-demo) on HuggingFace!

Traditional matching of MARC (Machine-Readable Cataloging) records has relied heavily on identifiers like OCLC numbers, ISBNs, LCCNs, etc. assigned by catalogers. However, this approach struggles with records having incorrect identifiers or lacking them altogether. This model has been developed to match MARC records based solely on their bibliographic metadata (title, author, publisher, etc.), enabling successful matches even when identifiers are missing or inaccurate.

We have primarily focused on MARC records for English monographs contributed to the HathiTrust by partnering institutions. The future direction of this repository is uncertain, but we plan to develop a new dataset and model encompassing a broader range of languages and publication locations. If you would like to contribute datasets, results, or models/methods, please contact us. We are eager to connect with others working on MARC record matching.

## Key Features
- Bibliographic Metadata Matching: Performs matching based solely on bibliographic data, eliminating the need for identifiers.
- Text Field Flexibility: Accommodates minor variations in bibliographic metadata fields for accurate matching.
- Adjustable Matching Threshold: Allows tuning the balance between false positives and false negatives based on specific use cases.

## Usage
The `marcai` package comes with a command-line interface offering a suite of commands for processing data, training models, and making predictions. All commands have their own help functions, which can be accessed by running `marc-ai <command> --help`.

### Processing data
`marc-ai process` takes a file containing MARC records and a CSV containing indices of record comparisons, and calculates similarity scores for several fields in the MARC records. These similarity values serve as the input features to the machine learning model.

### Training a model
`marc-ai train` trains a model with the hyperparameters defined in `config.yaml`, including the paths to dataset splits.

### Making predictions
`marc-ai predict` takes the output from `marc-ai process` and a trained model, and runs the similarity scores through the model to produce match confidence scores.

### Finding matches without I/O
`marc-ai find_matches` combines the commands for processing and predicting to cut out the unnecessary step of saving similarity values to disk. This is substantially faster when working with large amounts of data.

## Analysis (Jupyter Notebooks)

We have provided Jupyter notebooks containing analyses conducted during this project. The purpose of these notebooks is to examine the dataset and our current model results, as well as to share the methodologies employed throughout the project.

## Dataset

The initial dataset originates from HathiTrust contributors; however, the records have been anonymized, with identifiers and custom fields removed. This dataset was specifically designed to create and evaluate record pairing methods based on content alone, making it unsuitable for pairing records using both content and identifiers or evaluating the entire HathiTrust collection. The HathiTrust data is licensed under CC0, with certain caveats detailed in the [LICENSE.md](datasets/ht/LICENSE.md) file.

The data is real and may contain some errors and peculiarities due to the way HathiTrust combines monograph records. We plan to collaborate with HathiTrust to make this dataset more accessible to a wider audience, perhaps on Hugging Face. We welcome feedback on the format or any issues to improve its usefulness.

## Results

The results folder contains our model's outcomes, as well as some basic attempts at string matching and fuzzy string matching. These findings are used by the analysis notebooks to compare and contrast the performance of various methods.



