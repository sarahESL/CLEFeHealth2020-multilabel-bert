# clef2020eHealth-multilabel-bert
This repository includes the source code used in nlp4life@clef2020eHealth and fine-tuning BERT for multilabel sequence classification using HuggingFace Transformer.

## Requirements
### Data format
All input data must be in csv or tsv. Content of the input files must include `filename`, `text` and one column per ICD code to indicate weather the ICD code is associate to the corresponding `filename` or not. For example, assume in general we have three different ICD codes in our ground truth set:

| filename | text  | g45.9 | i08.1 | i72.2|
|----------| ----- | ----- | ----- | ---- |
|   file1  | text1 |   1   |   0   |   1  |

This means that `file1` is labeled with ICD10 codes `g45.9` and `i72.2`.
### Python dependencies
Dependcies exist in the `requirements.txt` file. Simply install them by `pip install -r requirements.txt`

## Data
* Pretrained fastText embedding from SBWC can be downloaded from [here](https://github.com/dccuchile/spanish-word-embeddings).
* The dataset from CLEFeHealth2020 challenge can be found at [here](https://zenodo.org/record/3758054#.Xwsikc8zY5l).
