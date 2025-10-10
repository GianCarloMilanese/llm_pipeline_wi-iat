# Investigating Large Language Models' Linguistic Abilities for Text Preprocessing

This repository contains the code for the experiments of the paper "Investigating
Large Language Models' Linguistic Abilities for Text Preprocessing"
submitted to WI-IAT.

The repository is organized as follows:

- `data`: the adopted datasets, split in train and test sets.
- `scripts`: the code used to run the experiments, described in detail below.
- `scripts/config`: the prompts passed to the LLMs to perform the preprocessing
  operations for each dataset.
- `scripts/datasets_preprocessed`: the outputs of the LLM-based preprocessing
  for each dataset.
- `scripts/results_analysis`: the scores related to the quality of the
  preprocessing performed by the LLMs (RQ1).
  The averages of these scores
  (grouped by language and model) are reported in the paper.

# Reproducing the experiments

## Libraries

Create a Python (>= 3.10) environment. The file `./requirements.txt` lists
the packages installed in the virtual environment used to run the experiments.
They can be installed by running.
```
pip install -r requirements.txt
```
In case of dependency or compatibility issues in running this command, the
following are the main libraries that should be installed:

- `accelerate`
- `datasets`
- `nltk`
- `numpy`
- `pandas`
- `protobuf`
- `scikit-learn`
- `sentencepiece`
- `spacy`
- `torch`
- `tqdm`
- `transformers`

Additionally, download a few spaCy and NLTK models and resources:
```
python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm
python -m spacy download es_core_news_sm
python -m spacy download fr_core_news_sm
python -m spacy download it_core_news_sm
python -m spacy download pt_core_news_sm
python ./scripts/preprocessing_utils.py # for NLTK stopwords and punkt
```

## Downloading and sampling the original datasets

The datasets are publicy available at: 
<a href="url">[Cardiff Tweet](https://huggingface.co/datasets/cardiffnlp/tweet_eval)</a> for English-written texts;
<a href="url">[Cardiff Multilingual](https://huggingface.co/datasets/cardiffnlp/tweet_sentiment_multilingual)</a> for Italian, French, German, Portuguese and Spanish datasets.

The script `./scripts/Create_dataset.py` generates the subsample used for our experiments.

## LLM-based preprocessing

The script `./scripts/llm_preprocessing.py` performs the LLM-based preprocessing
on a given dataset. An example of its usage is found in
`./scripts/run_preprocessing.sh`. The results of the preprocessing performed on
all the datasets in `./data` can be found in `./scripts/datasets_preprocessed/`.

## Analyzing the quality of the LLM-based preprocessing

The scripts
`./scripts/lemmatization_analysis.py`,
`./scripts/stemming_analysis.py`, and
`./scripts/stopwords_analysis.py`
analyze the quality of the text preprocessing performed by the LLMs.
The results are saved to `./scripts/results_analysis`.
The averages of these scores (grouped by language and model) reported in the
paper can be obtained by running `./scripts/analysis_averages.py`.

## Text classification

The script `./scripts/Optimizing_hyperparams.py` trains and evaluates the ML
algorithms based on traditional text preprocessing to find the optimal
hyperparameters.

The scripts `./scripts/Traditional_preprocessing_tc.py` and
`./scripts/LLM_preprocessing_tc.py` train and evaluate the ML text
classification algorithms on the traditionally preprocessed and LLM-preprocessed
text, respectively.
