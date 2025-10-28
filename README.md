# Hangman AI

This repository contains a Python implementation of an automated Hangman player that uses character-level n-gram language models to guess letters intelligently.

## Overview

The core of the project is `hangman_ai.py`, which includes:

- **hangman** – a function to simulate the Hangman game for a secret word and a guesser.
- A simple baseline **random_guesser** that selects an unused letter at random.
- A **unigram_guesser** that picks the most frequent unused letter, and a length-conditioned unigram variant.
- A **bigram_guesser** that looks at the previous letter to estimate the next one.
- A **bidirectional bigram** model (`my_amazing_ai_guesser`) that considers both left and right context.

Training functions build the necessary frequency tables from the Brown corpus. A `test_guesser` utility is provided to evaluate a guesser on a test set by averaging mistakes.

## Dataset

The module uses the NLTK Brown corpus to build language models. Words are lower-cased, filtered to alphabetic tokens and de-duplicated. The data is shuffled and split into a training set and a test set.

## Installation

Use pip to install the required packages:

```bash
pip install -r requirements.txt
```

The script will automatically download the Brown corpus the first time it runs.

## Usage

To build the models and evaluate each guesser on the Brown corpus, run:

```bash
python hangman_ai.py
```

This will print the average number of mistakes for the random baseline, unigram models, bigram model and bidirectional bigram model. You can also import the functions from `hangman_ai.py` into your own programs to play custom games or experiment with different data.

## Project structure

| File            | Description                                                                 |
|-----------------|-----------------------------------------------------------------------------|
| `hangman_ai.py` | Implementation of the Hangman game, language-model guessers and training utilities |
| `requirements.txt` | Lists dependencies (`numpy`, `nltk`)                                     |
| `README.md`     | Project description and instructions for installation and use               |

## License

Distributed under the MIT License.
