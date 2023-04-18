# MARC AI Project

This project aims to share data and code from our investigations into using neural networks for MARC data matching. We have primarily focused on MARC records for English monographs contributed to the HathiTrust by partnering institutions. The future direction of this repository is uncertain, but we plan to develop a new dataset and model encompassing a broader range of languages and publication locations over the next year. If you would like to contribute datasets, results, or models/methods to this repository, please contact us. We are eager to connect with others working on MARC record matching.

## Analysis (Jupyter Notebooks)

We have provided Jupyter notebooks containing analyses conducted during this project. The purpose of these notebooks is to examine the dataset and our current model results, as well as to share the methodologies employed throughout the project.

## Dataset

The initial dataset originates from HathiTrust contributors; however, the records have been anonymized, with identifiers and custom fields removed. This dataset was specifically designed to create and evaluate record pairing methods based on content alone, making it unsuitable for pairing records using both content and identifiers or evaluating the entire HathiTrust collection. The HathiTrust data is licensed under CC0, with certain caveats detailed in the [LICENSE.md](datasets/ht/LICENSE.md) file.

The data is real and may contain some errors and peculiarities due to the way HathiTrust combines monograph records. We plan to collaborate with HathiTrust to make this dataset more accessible to a wider audience, perhaps on Hugging Face. We welcome feedback on the format or any issues to improve its usefulness.

## Results

The results folder contains our model's outcomes, as well as some basic attempts at string matching and fuzzy string matching. These findings are used by the analysis notebooks to compare and contrast the performance of various methods.

## Model

We have yet to determine how or when we will share the code for the model, but if we share it anywhere we will post a link here.

