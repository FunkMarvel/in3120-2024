# pylint: disable=missing-module-docstring
# pylint: disable=line-too-long
import itertools
import math
from collections import Counter
from typing import Any, Dict, Iterable, Iterator

from . import Sieve
from .dictionary import InMemoryDictionary
from .normalizer import Normalizer
from .tokenizer import Tokenizer
from .corpus import Corpus


class NaiveBayesClassifier:
    """
    Defines a multinomial naive Bayes text classifier. For a detailed primer, see
    https://nlp.stanford.edu/IR-book/html/htmledition/naive-bayes-text-classification-1.html.
    """

    def __init__(self, training_set: Dict[str, Corpus], fields: Iterable[str],
                 normalizer: Normalizer, tokenizer: Tokenizer):
        """
        Trains the classifier from the named fields in the documents in the
        given training set.
        """
        # Used for breaking the text up into discrete classification features.
        self.__normalizer = normalizer
        self.__tokenizer = tokenizer

        # The vocabulary we've seen during training.
        self.__vocabulary = InMemoryDictionary()

        # Maps a category c to the logarithm of its prior probability,
        # i.e., c maps to log(Pr(c)).
        self.__priors: Dict[str, float] = {}

        # Maps a category c and a term t to the logarithm of its conditional probability,
        # i.e., (c, t) maps to log(Pr(t | c)).
        self.__conditionals: Dict[str, Dict[str, float]] = {}

        # Maps a category c to the denominator used when doing Laplace smoothing.
        self.__denominators: Dict[str, int] = {}

        # Train the classifier, i.e., estimate all probabilities.
        self.__compute_priors(training_set)
        self.__compute_vocabulary(training_set, fields)
        self.__compute_posteriors(training_set, fields)

    def __compute_priors(self, training_set) -> None:
        """
        Estimates all prior probabilities (or, rather, log-probabilities) needed for
        the naive Bayes classifier.
        """
        num_docs = sum(training_set[category].size() for category in training_set)

        for category in training_set:
            self.__priors[category] = math.log(training_set[category].size() / num_docs)

    def __compute_vocabulary(self, training_set, fields) -> None:
        """
        Builds up the overall vocabulary as seen in the training set.
        """
        for category in training_set:
            for document in training_set[category]: # iterable handling borrowed from invertedindex.py:
                all_terms = itertools.chain.from_iterable(self.__get_terms(document.get_field(f, "")) for f in fields)

                for term in all_terms:
                    self.__vocabulary.add_if_absent(term)

    def __compute_posteriors(self, training_set, fields) -> None:
        """
        Estimates all conditional probabilities (or, rather, log-probabilities) needed for
        the naive Bayes classifier.
        """
        for category in training_set: # populate conditionals with 1 smoothing occurrence per term:
            self.__conditionals[category] = {term: 1 for term, _ in self.__vocabulary}
            tot_term_occurrences = len(self.__vocabulary)

            for document in training_set[category]:  # iterable handling borrowed from invertedindex.py:
                all_terms = itertools.chain.from_iterable(self.__get_terms(document.get_field(f, "")) for f in fields)

                term_frequencies = Counter(all_terms)  # count actual occurrences
                tot_term_occurrences += sum(freq for freq in term_frequencies.values())

                for term, freq in term_frequencies.items():  # add up for each term
                    self.__conditionals[category][term] += freq

            # calculate log-probabilities for current class:
            for term, _ in self.__vocabulary:
                freq = self.__conditionals[category][term]
                self.__conditionals[category][term] = math.log(freq / tot_term_occurrences)



    def __get_terms(self, buffer) -> Iterator[str]:
        """
        Processes the given text buffer and returns the sequence of normalized
        terms as they appear. Both the documents in the training set and the buffers
        we classify need to be identically processed.
        """
        tokens = self.__tokenizer.strings(self.__normalizer.canonicalize(buffer))
        return (self.__normalizer.normalize(t) for t in tokens)

    def get_prior(self, category: str) -> float:
        """
        Given a category c, returns the category's prior log-probability log(Pr(c)).

        This is an internal detail having public visibility to facilitate testing.
        """
        return self.__priors[category]

    def get_posterior(self, category: str, term: str) -> float:
        """
        Given a category c and a term t, returns the posterior log-probability log(Pr(t | c)).

        This is an internal detail having public visibility to facilitate testing.
        """
        assert category in self.__conditionals
        assert term in self.__conditionals[category]

        return self.__conditionals[category][term]

    def classify(self, buffer: str) -> Iterator[Dict[str, Any]]:
        """
        Classifies the given buffer according to the multinomial naive Bayes rule. The computed (score, category) pairs
        are emitted back to the client via the supplied callback sorted according to the scores. The reported scores
        are log-probabilities, to minimize numerical underflow issues. Logarithms are base e.

        The results yielded back to the client are dictionaries having the keys "score" (float) and
        "category" (str).
        """
        scores = Sieve(len(self.__conditionals))
        all_terms = self.__get_terms(buffer)
        term_frequencies = Counter(all_terms)

        for category, conditionals in self.__conditionals.items():
            score = self.__priors[category]
            for term, freq in term_frequencies.items():
                if term not in self.__vocabulary:
                    continue
                score += conditionals[term]*freq

            scores.sift(score, category)

        yield from ({"score": score, "category": cat} for score, cat in scores.winners())

# example run of assignments.py e-1:
r"""
(venv) PS E:\Documents\in3120-2024\tests> python.exe .\assignments.py e-1
test_china_example_from_textbook (test_naivebayesclassifier.TestNaiveBayesClassifier.test_china_example_from_textbook) ... ok
test_language_detection_trained_on_some_news_corpora (test_naivebayesclassifier.TestNaiveBayesClassifier.test_language_detection_trained_on_some_news_corpora) ... ok
test_predict_movie_genre_from_movie_title (test_naivebayesclassifier.TestNaiveBayesClassifier.test_predict_movie_genre_from_movie_title) ... ok
test_predict_name_of_search_engine_from_description (test_naivebayesclassifier.TestNaiveBayesClassifier.test_predict_name_of_search_engine_from_description) ... ok
test_scores_are_sorted_descending (test_naivebayesclassifier.TestNaiveBayesClassifier.test_scores_are_sorted_descending) ... ok
test_uses_yield (test_naivebayesclassifier.TestNaiveBayesClassifier.test_uses_yield) ... ok

----------------------------------------------------------------------
Ran 6 tests in 1.692s

OK
"""

# example run of repl.py e-1:
r"""
(venv) PS E:\Documents\in3120-2024\tests> python.exe .\repl.py e-1       
Initializing naive Bayes classifier from news corpora...
Enter some text and classify it into ['en', 'no', 'da', 'de'].
Returned scores are log-probabilities.
Ctrl-C to exit.
text>Seks mine jarlar heime vera, gøyme det gullet balde, andre seks på heidningalando då svinga dei jørni kalde!
[{'category': 'no', 'score': -98.75397295627718},
 {'category': 'da', 'score': -106.06572984644481},
 {'category': 'de', 'score': -136.14885654039784},
 {'category': 'en', 'score': -136.317660020898}]
Evaluation took 0.0002134000005753478 seconds.
text>
Bye!
"""