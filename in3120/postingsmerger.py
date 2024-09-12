# pylint: disable=missing-module-docstring

from typing import Iterator
from .posting import Posting


class PostingsMerger:
    """
    Utility class for merging posting lists.

    It is currently left unspecified what to do with the term frequency field
    in the returned postings when document identifiers overlap. Different
    approaches are possible, e.g., an arbitrary one of the two postings could
    be returned, or the posting having the smallest/largest term frequency, or
    a new one that produces an averaged value, or something else.

    Note that the result of merging posting lists is itself a posting list.
    Hence the merging methods can be combined to compute the result of more
    complex Boolean operations over posting lists.
    """

    @staticmethod
    def intersection(iter1: Iterator[Posting], iter2: Iterator[Posting]) -> Iterator[Posting]:
        """
        A generator that yields a simple AND(A, B) of two posting
        lists A and B, given iterators over these.

        In set notation, this corresponds to computing the intersection
        D(A) ∩ D(B), where D(A) and D(B) are the sets of documents that
        appear in A and B: A posting appears once in the result if and
        only if the document referenced by the posting appears in both
        D(A) and D(B).

        The posting lists are assumed sorted in increasing order according
        to the document identifiers.
        """
        posting1 = next(iter1, None)  # don't know if this counts as temporary data structures,
        posting2 = next(iter2, None)  # but didn't figure out how to do the selective iteration
                                      # without the posting1 and posting2 temp references.

        # would appreciate some feedback on writing proper/more pythonic code,
        # as my background is C++ and C#.
        while posting1 is not None and posting2 is not None:
            if posting1.document_id == posting2.document_id:
                yield Posting(posting1.document_id, min(posting1.term_frequency, posting2.term_frequency))
                posting1 = next(iter1, None)
                posting2 = next(iter2, None)
            elif posting1.document_id < posting2.document_id:
                posting1 = next(iter1, None)
            else:
                posting2 = next(iter2, None)


    @staticmethod
    def union(iter1: Iterator[Posting], iter2: Iterator[Posting]) -> Iterator[Posting]:
        """
        A generator that yields a simple OR(A, B) of two posting
        lists A and B, given iterators over these.

        In set notation, this corresponds to computing the union
        D(A) ∪ D(B), where D(A) and D(B) are the sets of documents that
        appear in A and B: A posting appears once in the result if and
        only if the document referenced by the posting appears in either
        D(A) or D(B).

        The posting lists are assumed sorted in increasing order according
        to the document identifiers.
        """
        posting1 = next(iter1, None)
        posting2 = next(iter2, None)

        while posting1 is not None or posting2 is not None:
            if posting2 is None or (posting1 is not None and posting1.document_id < posting2.document_id):
                yield posting1
                posting1 = next(iter1, None)
            elif posting1 is not None and posting1.document_id == posting2.document_id:
                yield Posting(posting1.document_id, max(posting1.term_frequency, posting2.term_frequency))
                posting1 = next(iter1, None)
                posting2 = next(iter2, None)
            else:
                yield posting2
                posting2 = next(iter2, None)



        # raise NotImplementedError("You need to implement this as part of the obligatory assignment.")

    @staticmethod
    def difference(iter1: Iterator[Posting], iter2: Iterator[Posting]) -> Iterator[Posting]:
        """
        A generator that yields a simple ANDNOT(A, B) of two posting
        lists A and B, given iterators over these.

        In set notation, this corresponds to computing the difference
        D(A) - D(B), where D(A) and D(B) are the sets of documents that
        appear in A and B: A posting appears once in the result if and
        only if the document referenced by the posting appears in D(A)
        but not in D(B).

        The posting lists are assumed sorted in increasing order according
        to the document identifiers.
        """
        posting1 = next(iter1, None)
        posting2 = next(iter2, None)

        while posting1 is not None:
            if posting2 is None or posting1.document_id < posting2.document_id:
                yield posting1
                posting1 = next(iter1, None)
            elif posting1.document_id == posting2.document_id:
                posting1 = next(iter1, None)
                posting2 = next(iter2, None)
            else:
                posting2 = next(iter2, None)

# example run of repl.py a-2:
r"""
(venv) PS E:\Documents\in3120-2024\tests> python.exe .\repl.py a-2
Building inverted index from English name corpus...
Enter a complex Boolean query expression and find matching documents.
Lookup options are {'optimize': True}.
Ctrl-C to exit.
query>AND(Alexander, OR(Davis, Pratt))
[{'document': {'document_id': 1968, 'fields': {'body': 'Alexander Davis'}}},
 {'document': {'document_id': 2667, 'fields': {'body': 'Alexander Pratt'}}}]
Evaluation took 0.00024259999918285757 seconds.
query>
"""