"""
Hangman AI using Character n‑Gram Models
---------------------------------------

This module implements a series of automated guessers for the
traditional Hangman word game. Each guesser employs a different
language model to predict the next character based on the current state
of the game, ranging from a simple random baseline to unigram and
bigram models, as well as a more sophisticated bidirectional bigram
model. The code is designed to be self contained and free of
interactive prompts so that it can be imported as a module or run from
the command line for experimentation.

The main components are:

* **hangman** – core game loop that evaluates a guesser on a given
  secret word.
* **random_guesser** – baseline method that picks an unused letter at
  random.
* **unigram_guesser** – chooses the most frequent letter in the
  training corpus not yet guessed.
* **unigram_length_guesser** – conditions the unigram model on the
  length of the secret word.
* **bigram_guesser** – uses a character bigram model with fallback to
  the unigram model when no context is available.
* **my_amazing_ai_guesser** – a bidirectional bigram approach that
  considers both left and right contexts for masked positions.

Training functions are provided to build the required frequency
dictionaries from the NLTK Brown corpus. A ``test_guesser`` utility is
also included to compute the average number of mistakes made by a
guesseer over a test set.
"""

from __future__ import annotations

import random
import re
from collections import Counter, defaultdict
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import nltk
from nltk.corpus import brown


def hangman(
    secret_word: str,
    guesser: Callable[[List[str], Sequence[str]], str],
    max_mistakes: int = 8,
    verbose: bool = False,
    **guesser_args,
) -> int:
    """Play a game of hangman with a given guesser.

    Parameters
    ----------
    secret_word: str
        The hidden word to guess. It will be lowercased internally.
    guesser: callable
        Function that takes the current mask (list of characters with
        '_' for unknowns) and a sequence of guessed characters, and
        returns the next guess.
    max_mistakes: int
        Maximum number of incorrect guesses allowed before the game ends.
    verbose: bool
        If True, prints progress information.
    guesser_args: dict
        Additional keyword arguments passed directly to the guesser.

    Returns
    -------
    int
        The number of incorrect guesses made.
    """
    secret_word = secret_word.lower()
    mask = ['_'] * len(secret_word)
    guessed: set[str] = set()
    mistakes = 0
    if verbose:
        print("Starting hangman game. Target is", ' '.join(mask), 'length', len(secret_word))
    while mistakes < max_mistakes:
        if verbose:
            print("You have", (max_mistakes - mistakes), "attempts remaining.")
        guess = guesser(mask, guessed, **guesser_args)
        if verbose:
            print('Guess is', guess)
        if guess in guessed:
            if verbose:
                print('Already guessed this before.')
            mistakes += 1
        else:
            guessed.add(guess)
            if guess in secret_word and len(guess) == 1:
                for i, c in enumerate(secret_word):
                    if c == guess:
                        mask[i] = c
                if verbose:
                    print('Good guess:', ' '.join(mask))
            else:
                if verbose:
                    print('Sorry, try again.')
                mistakes += 1
        if '_' not in mask:
            if verbose:
                print('Congratulations, you won.')
            return mistakes
    if verbose:
        print('Out of guesses. The word was', secret_word)
    return mistakes


def test_guesser(guesser: Callable[[List[str], Sequence[str]], str], test_words: Iterable[str], **guesser_args) -> float:
    """Evaluate a guesser by averaging mistakes over a test set."""
    total = 0
    for word in test_words:
        total += hangman(word, guesser, max_mistakes=26, verbose=False, **guesser_args)
    return total / float(len(list(test_words)))


def random_guesser(mask: List[str], guessed: Sequence[str], **kwargs) -> str:
    """Return a random letter that hasn't been guessed yet."""
    alphabet = [chr(ord('a') + i) for i in range(26)]
    choices = [c for c in alphabet if c not in guessed]
    return random.choice(choices)


def build_unigram_counts(words: Iterable[str]) -> Counter:
    """Construct a Counter of character frequencies from a list of words."""
    counts: Counter = Counter()
    for w in words:
        counts.update(w)
    return counts


def unigram_guesser(mask: List[str], guessed: Sequence[str], unigram_counts: Counter) -> str:
    """Guess the most frequent unused letter under a unigram model."""
    total = sum(unigram_counts.values())
    probs = {c: unigram_counts[c] / total for c in unigram_counts}
    alphabet = [chr(ord('a') + i) for i in range(26)]
    choices = [c for c in alphabet if c not in guessed]
    return max(choices, key=lambda x: probs.get(x, 0))


def build_unigram_counts_by_length(words: Iterable[str]) -> Dict[int, Counter]:
    """Build unigram frequency tables keyed by word length."""
    counts: Dict[int, Counter] = {}
    for w in words:
        l = len(w)
        counts.setdefault(l, Counter())
        counts[l].update(w)
    return counts


def unigram_length_guesser(
    mask: List[str], guessed: Sequence[str], *, counts_by_length: Dict[int, Counter], fallback_counts: Counter
) -> str:
    """Guess using a length‑conditioned unigram model with fallback."""
    length = len(mask)
    counts = counts_by_length.get(length, fallback_counts)
    total = sum(counts.values())
    probs = {c: counts[c] / total for c in counts}
    alphabet = [chr(ord('a') + i) for i in range(26)]
    choices = [c for c in alphabet if c not in guessed]
    return max(choices, key=lambda x: probs.get(x, 0))


def build_bigram_counts(words: Iterable[str]) -> Dict[str, Counter]:
    """Create a bigram frequency dictionary from training words.

    A start symbol ``$`` is prepended to each word to model word‑initial
    probabilities.
    """
    bigram_counts: Dict[str, Counter] = defaultdict(Counter)
    for word in words:
        padded = ['$'] + list(word)
        for a, b in zip(padded[:-1], padded[1:]):
            bigram_counts[a][b] += 1
    return bigram_counts


def bigram_guesser(
    mask: List[str],
    guessed: Sequence[str],
    *,
    bigram_counts: Dict[str, Counter],
    unigram_counts: Counter,
) -> str:
    """Guess using a bigram model, falling back to the unigram model when no context is available."""
    alphabet = [chr(ord('a') + i) for i in range(26)]
    guessed_set = set(guessed)
    bigram_probs: Dict[str, Dict[str, float]] = {}
    for prev, counter in bigram_counts.items():
        total = sum(counter.values())
        bigram_probs[prev] = {c: counter[c] / total for c in counter}
    total_uni = sum(unigram_counts.values())
    unigram_probs = {c: unigram_counts[c] / total_uni for c in unigram_counts}
    guess_scores = {c: 0.0 for c in alphabet}
    extended_mask = ['$'] + mask
    for i, ch in enumerate(extended_mask):
        if ch == '_':
            prev = extended_mask[i - 1] if i > 0 else '$'
            for c in alphabet:
                if c in guessed_set:
                    continue
                prob = bigram_probs.get(prev, {}).get(c, None)
                if prob is None:
                    prob = unigram_probs.get(c, 0)
                guess_scores[c] += prob
    return max((c for c in alphabet if c not in guessed_set), key=lambda x: guess_scores[x])


def build_bidirectional_bigram_counts(words: Iterable[str]) -> Tuple[Dict[str, Counter], Dict[str, Counter], Dict[Tuple[str, str], Counter], Counter]:
    """Compute frequency tables for a bidirectional bigram model.

    Returns
    -------
    forward : dict
        Standard forward bigram counts.
    backward : dict
        Reverse bigram counts keyed by next character.
    bidi : dict
        Counts keyed by (left_context, right_context) for the centre character.
    unigram : Counter
        Overall unigram counts.
    """
    forward: Dict[str, Counter] = defaultdict(Counter)
    backward: Dict[str, Counter] = defaultdict(Counter)
    bidi: Dict[str, Counter] = defaultdict(Counter)
    uni: Counter = Counter()
    for word in words:
        padded = ['$'] + list(word) + ['/']
        uni.update(word)
        for a, b in zip(padded[:-1], padded[1:]):
            forward[a][b] += 1
        for i in range(1, len(padded) - 1):
            backward[padded[i + 1]][padded[i]] += 1
        for i in range(1, len(padded) - 1):
            centre = padded[i]
            left = padded[i - 1]
            right = padded[i + 1]
            bidi[centre][(left, right)] += 1
    return forward, backward, bidi, uni


def my_amazing_ai_guesser(
    mask: List[str],
    guessed: Sequence[str],
    *,
    forward: Dict[str, Counter],
    backward: Dict[str, Counter],
    bidi: Dict[str, Counter],
    unigram: Counter,
) -> str:
    """Bidirectional bigram guesser combining left and right context."""
    alphabet = [chr(ord('a') + i) for i in range(26)]
    guessed_set = set(guessed)
    total_uni = sum(unigram.values())
    uni_probs = {c: unigram[c] / total_uni for c in unigram}
    fwd_probs: Dict[str, Dict[str, float]] = {
        prev: {c: cnt / sum(counter.values()) for c, cnt in counter.items()}
        for prev, counter in forward.items()
    }
    bwd_probs: Dict[str, Dict[str, float]] = {
        nxt: {c: cnt / sum(counter.values()) for c, cnt in counter.items()}
        for nxt, counter in backward.items()
    }
    bidi_probs: Dict[str, Dict[Tuple[str, str], float]] = {
        c: {ctx: cnt / sum(counter.values()) for ctx, cnt in counter.items()}
        for c, counter in bidi.items()
    }
    extended = ['$'] + mask + ['/']
    scores = {c: 0.0 for c in alphabet}
    for i, ch in enumerate(extended):
        if ch == '_':
            left = extended[i - 1]
            right = extended[i + 1]
            for c in alphabet:
                if c in guessed_set:
                    continue
                score = 0.0
                if (left, right) in bidi_probs.get(c, {}):
                    score = bidi_probs[c][(left, right)]
                else:
                    if c in fwd_probs.get(left, {}):
                        score = fwd_probs[left][c]
                    elif c in bwd_probs.get(right, {}):
                        score = bwd_probs[right][c]
                    else:
                        score = uni_probs.get(c, 0)
                scores[c] += score
    return max((c for c in alphabet if c not in guessed_set), key=lambda x: scores[x])


def prepare_dataset(seed: int = 1, test_size: int = 1000) -> Tuple[List[str], List[str]]:
    """Load the Brown corpus, filter to alphabetic words, shuffle and split."""
    nltk.download('brown')
    words = [w.lower() for w in brown.words() if re.fullmatch(r"[a-zA-Z]+", w)]
    unique = list(set(words))
    np.random.seed(seed)
    np.random.shuffle(unique)
    test_set = unique[:test_size]
    training_set = unique[test_size:]
    return training_set, test_set


if __name__ == "__main__":
    train_words, test_words = prepare_dataset(seed=1, test_size=1000)
    uni_counts = build_unigram_counts(train_words)
    uni_by_len = build_unigram_counts_by_length(train_words)
    bi_counts = build_bigram_counts(train_words)
    fwd, bwd, bidi_counts, uni_counts_adv = build_bidirectional_bigram_counts(train_words)
    print("Random baseline:", test_guesser(random_guesser, test_words))
    print("Unigram:", test_guesser(lambda m, g: unigram_guesser(m, g, uni_counts), test_words))
    print(
        "Unigram with length:",
        test_guesser(
            lambda m, g: unigram_length_guesser(m, g, counts_by_length=uni_by_len, fallback_counts=uni_counts),
            test_words,
        ),
    )
    print(
        "Bigram:",
        test_guesser(
            lambda m, g: bigram_guesser(m, g, bigram_counts=bi_counts, unigram_counts=uni_counts), test_words
        ),
    )
(  print(
        "Bidirectional bigram:",
        test_guesser(
            lambda m, g: my_amazing_ai_guesser(
                m, g, forward=fwd, backward=bwd, bidi=bidi_counts, unigram=uni_counts_adv
            ),
            test_words,
        ),
    )
