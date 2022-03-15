"""
LCM: Linear time Closed item set Miner
as described in `http://lig-membres.imag.fr/termier/HLCM/hlcm.pdf`
"""

# Authors: Rémi Adon <remi.adon@gmail.com>
#          Luis Galárraga <galarraga@luisgalarraga.de>
#
# License: BSD 3 clause

from collections import defaultdict
from itertools import takewhile
from typing import Union, List, Dict, Any, Tuple, FrozenSet, Generator, Iterable, Optional, Set

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pandas import DataFrame
from sortedcontainers import SortedDict, SortedKeyList


from ..utils import _check_min_supp
from ..utils import filter_maximal
from ..bitmaps import Bitmap

from ..base import BaseMiner, DiscovererMixin


class LCM(BaseMiner, DiscovererMixin):
    """
    Linear time Closed item set Miner.

    LCM can be used as a **generic purpose** miner, yielding some patterns
    that will be later submitted to a custom acceptance criterion.

    It can also be used to simply discover the set of **closed itemsets** from
    a transactional dataset.

    Parameters
    ----------
    min_supp: int or float, default=0.2
        The minimum support for itemsets to be rendered in the output
        Either an int representing the absolute support, or a float for relative support
        Default to 0.2 (20%)

    max_depth: int, default=20
        Maximum depth for exploration in the search space.
        When going into recursion, we check if the current depth
        is **strictly greater** than `max_depth`.
        If this is the case, we stop.
        This can avoid cumbersome computation.
        A **root node is considered of depth 0**.

    n_jobs : int, default=1
        The number of jobs to use for the computation. Each single item is attributed a job
        to discover potential itemsets, considering this item as a root in the search space.
        **Processes are preffered** over threads.

    References
    ----------
    .. [1]
        Takeaki Uno, Masashi Kiyomi, Hiroki Arimura
        "LCM ver. 2: Efficient mining algorithms for frequent/closed/maximal itemsets", 2004

    .. [2] Alexandre Termier
        "Pattern mining rock: more, faster, better"

    Examples
    --------

    >>> from skmine.itemsets import LCM
    >>> from skmine.datasets.fimi import fetch_chess
    >>> chess = fetch_chess()
    >>> lcm = LCM(min_supp=2000)
    >>> patterns = lcm.fit_discover(chess)      # doctest: +SKIP
    >>> patterns.head()                         # doctest: +SKIP
        itemset support
    0      (58)    3195
    1  (11, 58)    2128
    2  (15, 58)    2025
    3  (17, 58)    2499
    4  (21, 58)    2224
    >>> patterns[patterns.itemset.map(len) > 3]  # doctest: +SKIP
    """

    def __init__(self, *, min_supp=0.2, max_depth=20, n_jobs=1, verbose=False):
        _check_min_supp(min_supp)
        self.min_supp = min_supp  # provided by user
        self.max_depth = int(max_depth)
        self._min_supp = _check_min_supp(self.min_supp)
        self.itemid_to_tids_ = SortedDict()

        # Be aware that IDs for item start at 1.
        self.itemid_to_item: Dict[int, Item] = dict()
        self.item_to_itemid: Dict[Item, int] = dict()

        self.n_transactions_ = 0
        self.ctr = 0
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, D, y=None):
        """
        fit LCM on the transactional database, by keeping records of singular items
        and their transaction ids.

        Parameters
        ----------
        D: pd.Series or iterable
            a transactional database. All entries in this D should be lists.
            If D is a pandas.Series, then `(D.map(type) == list).all()` should return `True`

        Raises
        ------
        TypeError
            if any entry in D is not iterable itself OR if any item is not **hashable**
            OR if all items are not **comparable** with each other.
        """
        self.n_transactions_ = 0  # reset for safety
        item_to_tids = defaultdict(Bitmap)
        for transaction in D:
            for item in transaction:
                item_to_tids[item].add(self.n_transactions_)
            self.n_transactions_ += 1

        if isinstance(self.min_supp, float):
            # make support absolute if needed
            self._min_supp = self.min_supp * self.n_transactions_

        # Removes items with low support
        low_supp_items = [k for k, v in item_to_tids.items() if len(v) < self._min_supp]
        for item in low_supp_items:
            del item_to_tids[item]

        # Sort the item's transactions dictionary such that the item with the most support is 1st.
        supp_sorted_items = sorted(
            item_to_tids.items(), key=lambda e: len(e[1]), reverse=True
        )

        # For each item, replace its name by the ID of frequency (the more frequent an item is, the lower is its ID).
        for item_id, (item, transactions) in enumerate(supp_sorted_items):
            # Register the itemid and its transactions in the SortedDict to be used later.
            self.itemid_to_tids_[item_id + 1] = transactions

            # Save the information about which ID correspond to which item (in both direction).
            self.item_to_itemid[item] = item_id + 1
            self.itemid_to_item[item_id + 1] = item

        return self

    def discover(self, *, return_tids=False, return_depth=False):
        """Return the set of closed itemsets, with respect to the minium support

        Parameters
        ----------
        D : pd.Series or Iterable
            The input transactional database
            Where every entry contain singular items
            Items must be both hashable and comparable

        return_tids: bool, default=False
            Either to return transaction ids along with itemset.
            Default to False, will return supports instead

        return_depth: bool, default=False
            Either to return depth for each item or not.

        Returns
        -------
        pd.DataFrame
            DataFrame with the following columns
                ==========  =================================
                itemset     a `tuple` of co-occured items
                support     frequence for this itemset
                ==========  =================================

            if `return_tids=True` then
                ==========  =================================
                itemset     a `tuple` of co-occured items
                tids        a bitmap tracking positions
                ==========  =================================

            if `return_depth` is `True`, then a `depth` column is also present

        Example
        -------
        >>> from skmine.itemsets import LCM
        >>> D = [[1, 2, 3, 4, 5, 6], [2, 3, 5], [2, 5]]
        >>> LCM(min_supp=2).fit_discover(D)
             itemset  support
        0     (2, 5)        3
        1  (2, 3, 5)        2
        >>> LCM(min_supp=2).fit_discover(D, return_tids=True, return_depth=True) # doctest: +SKIP
             itemset       tids depth
        0     (2, 5)  [0, 1, 2]     0
        1  (2, 3, 5)     [0, 1]     1
        """

        dfs = Parallel(n_jobs=self.n_jobs, prefer="processes")(
            delayed(self._explore_root)(item, tids) for item, tids in self.itemid_to_tids_.items()
        )

        # make sure we have something to concat
        dfs.append(pd.DataFrame(columns=["itemset", "tids", "depth"]))
        df = pd.concat(dfs, axis=0, ignore_index=True)
        if not return_tids:
            df.loc[:, "support"] = df["tids"].map(len).astype(np.uint32)
            df.drop("tids", axis=1, inplace=True)

        if not return_depth:
            df.drop("depth", axis=1, inplace=True)
        return df

    def _explore_root(self, item, tids):
        it = self._inner((frozenset(), tids), item)
        df = pd.DataFrame(data=it, columns=["itemset", "tids", "depth"])
        if self.verbose and not df.empty:
            print("LCM found {} new itemsets from item : {}".format(len(df), item))
        return df

    def _inner(self, p_tids, limit, depth=0):
        if depth >= self.max_depth:
            return
        p, tids = p_tids
        # project and reduce DB w.r.t P
        cp = (
            item
            for item, ids in reversed(self.itemid_to_tids_.items())
            if tids.issubset(ids)
            if item not in p
        )

        # items are in reverse order, so the first consumed is the max
        max_k = next(takewhile(lambda e: e >= limit, cp), None)

        if max_k and max_k == limit:
            p_prime = (
                p | set(cp) | {max_k}
            )  # max_k has been consumed when calling next()

            # Translate the item IDs into their str for pretty printed patterns.
            p_prime_str: List[Item] = list(map(lambda itemid: self.itemid_to_item[itemid], p_prime))
            yield tuple(sorted(p_prime_str)), tids, depth

            candidates = self.itemid_to_tids_.keys() - p_prime
            candidates = candidates[: candidates.bisect_left(limit)]
            for new_limit in candidates:
                ids = self.itemid_to_tids_[new_limit]
                if tids.intersection_len(ids) >= self._min_supp:
                    # new pattern and its associated tids
                    new_p_tids = (p_prime, tids.intersection(ids))
                    yield from self._inner(new_p_tids, new_limit, depth + 1)


# Just for type comprehension.
Item = Any
Pattern = FrozenSet


class LCMNeighbours(BaseMiner, DiscovererMixin):
    """
    Customized version of LCM (described above) to include the notion of neighbour patterns in the support computation.

    Parameters
    ----------
    item_to_neighbours: Dict[Item, List[Item]]
        The neighbours of each items, used for the neighbour pattern generation.
        If the dictionnary is empty, LCMNeighbours is equivalent to LCM without the neighbours notion.

    min_supp: int or float, default=0.2
        The minimum support for itemsets to be rendered in the output, while taking into account the neighbours.
        The pattern returns may have a true support lower than this if thay are "supported" by neighbour patterns alone.
        Either an int representing the absolute support, or a float for relative support
        Default to 0.2 (20%)

    max_depth: int, default=20
        Maximum depth for exploration in the search space.
        When going into recursion, we check if the current depth
        is **strictly greater** than `max_depth`.
        If this is the case, we stop.
        This can avoid cumbersome computation.
        A **root node is considered of depth 0**.

    n_jobs : int, default=1
        The number of jobs to use for the computation. Each single item is attributed a job
        to discover potential itemsets, considering this item as a root in the search space.
        **Processes are preffered** over threads.

    References
    ----------
    .. [1]
        Takeaki Uno, Masashi Kiyomi, Hiroki Arimura
        "LCM ver. 2: Efficient mining algorithms for frequent/closed/maximal itemsets", 2004

    .. [2] Alexandre Termier
        "Pattern mining rock: more, faster, better"

    """

    def __init__(self, *, min_supp=0.2, max_depth=20, n_jobs=1, verbose=False):
        _check_min_supp(min_supp)
        self.min_supp: Union[int, float] = min_supp  # provided by user
        self.max_depth: int = int(max_depth)
        self._min_supp: Union[int, float] = _check_min_supp(self.min_supp)

        # Holds the information about in which transaction is present each item.
        self.itemid_to_tids_: SortedDict[Item, Bitmap] = SortedDict()

        # Be aware that IDs for item start at 1.
        self.itemid_to_item: Dict[int, Item] = dict()
        self.item_to_itemid: Dict[Item, int] = dict()

        # Total number of transactions.
        self.n_transactions_: int = 0

        # Holds the information about the neighbours of each item.
        self._item_to_neighbours: Dict[int, Set] = defaultdict(set)

        self.ctr = 0
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, D, item_to_neighbours: Dict[Item, List[Item]]=defaultdict(list), y=None):
        """
        Fit LCMNeighbours on the transactional database, by keeping records of singular items
        and their transaction ids.

        Parameters
        ----------
        D: pd.Series or iterable
            a transactional database. All entries in this D should be lists.
            If D is a pandas.Series, then `(D.map(type) == list).all()` should return `True`

        Raises
        ------
        TypeError
            if any entry in D is not iterable itself OR if any item is not **hashable**
            OR if all items are not **comparable** with each other.
        """
        self.n_transactions_ = 0  # reset for safety

        # Initializes the dictionary holding the information about in which transaction is present each item.
        item_to_tids: Dict[Any, Bitmap] = defaultdict(Bitmap)
        for transaction in D:  # For all transactions
            for item in transaction:  # For all items in the current transaction
                # Add the n° of current transaction in the list of transaction for the current item.
                item_to_tids[item].add(self.n_transactions_)
            self.n_transactions_ += 1  # Update n° of transaction for the next one.

        if isinstance(self.min_supp, float):
            # make support absolute if needed
            self._min_supp = self.min_supp * self.n_transactions_

        # Sort the item's transactions dictionary such that the item with the most support is 1st.
        supp_sorted_items = sorted(
            item_to_tids.items(), key=lambda e: len(e[1]), reverse=True
        )

        # For each item, replace its name by the ID of frequency (the more frequent an item is, the lower is its ID).
        for item_id, (item, transactions) in enumerate(supp_sorted_items):
            # Register the itemid and its transactions in the SortedDict to be used later.
            self.itemid_to_tids_[item_id+1] = transactions

            # Save the information about which ID correspond to which item (in both direction).
            self.item_to_itemid[item] = item_id+1
            self.itemid_to_item[item_id+1] = item

        # Translate the neighbours to have itemid instead of raw items.
        for item, lst_neighbours in item_to_neighbours.items():
            item_id = self.item_to_itemid[item]
            for neighbour in lst_neighbours:
                self._item_to_neighbours[item_id].add(self.item_to_itemid[neighbour])

        return self

    def discover(self, *, return_tids=False, return_depth=False):
        """Return the set of closed itemsets, with respect to the minimum support (while taking into account neighbour
        patterns).

        Parameters
        ----------
        D : pd.Series or Iterable
            The input transactional database
            Where every entry contain singular items
            Items must be both hashable and comparable

        return_tids: bool, default=False
            Either to return transaction ids along with itemset.
            Default to False, will return supports instead

        return_depth: bool, default=False
            Either to return depth for each item or not.

        Returns
        -------
        pd.DataFrame
            DataFrame with the following columns
                ==========  =================================
                itemset     a `tuple` of co-occured items
                support     frequence for this itemset
                ==========  =================================

            if `return_tids=True` then
                ==========  =================================
                itemset     a `tuple` of co-occured items
                tids        a bitmap tracking positions
                ==========  =================================

            if `return_depth` is `True`, then a `depth` column is also present

        """

        # Explore the tree of patterns for each single item and knowing in which transaction they are in.
        # Only for the itemsets with a high enough support.
        dfs = Parallel(n_jobs=self.n_jobs, prefer="processes")(
            delayed(self._explore_root)(item, tids) for item, tids in self.itemid_to_tids_.items()
            if self._is_neighbourfrequent(frozenset([item]), self._min_supp)
        )

        # make sure we have something to concat
        dfs.append(pd.DataFrame(columns=["itemset", "tids", "depth"]))
        df = pd.concat(dfs, axis=0, ignore_index=True)
        if not return_tids:
            df.loc[:, "support"] = df["tids"].map(len).astype(np.uint32)
            df.drop("tids", axis=1, inplace=True)

        if not return_depth:
            df.drop("depth", axis=1, inplace=True)
        return df

    def _explore_root(self, item: Item, tids: Bitmap) -> DataFrame:
        # Mines the pattern of the given item.
        it = self._inner((frozenset(), tids), item)

        # Recovers the patterns to create the result DataFrame.
        df = pd.DataFrame(data=it, columns=["itemset", "tids", "depth"])

        if self.verbose and not df.empty:
            print("LCM found {} new itemsets from item : {}".format(len(df), item))
        return df

    def _inner(self, p_tids: Tuple[Pattern, Bitmap], limit: Item, depth=0):

        # If we reached the maximum depth, stop the pattern generation.
        if depth >= self.max_depth:
            return

        pattern, pattern_transaction_ids = p_tids

        # project and reduce DB w.r.t P
        cp: Generator[Item] = (
            item
            for item, item_transction_ids in reversed(self.itemid_to_tids_.items())
            if pattern_transaction_ids.issubset(item_transction_ids)
            if item not in pattern
        )

        # items are in reverse order, so the first consumed is the max
        max_k = next(takewhile(lambda e: e >= limit, cp), None)

        if max_k is not None and max_k == limit:
            p_prime: Pattern = (
                pattern | set(cp) | {max_k}
            )  # max_k has been consumed when calling next()

            # If the current pattern
            # sorted items in ouput for better reproducibility
            if self._is_neighbourfrequent(p_prime, self._min_supp):

                p_prime_str: List[Item] = list(map(lambda itemid: self.itemid_to_item[itemid], p_prime))
                yield tuple(sorted(p_prime_str)), pattern_transaction_ids, depth

                candidates: Iterable = self.itemid_to_tids_.keys() - p_prime
                candidates = candidates[: candidates.bisect_left(limit)]
                for new_limit in candidates:
                    ids = self.itemid_to_tids_[new_limit]
                    candidate_pattern = p_prime | frozenset([new_limit])

                    # Check if the candidate pattern with its new limit would be frequent or not.
                    if self._is_neighbourfrequent(candidate_pattern, self._min_supp):
                        # new pattern and its associated tids
                        new_p_tids = (p_prime, pattern_transaction_ids.intersection(ids))
                        yield from self._inner(new_p_tids, new_limit, depth + 1)

    def _is_neighbourfrequent(self, pattern: Pattern, min_supp: Optional[int] = None) -> bool:
        """Computes whether a pattern is expected to be frequent while taking into account its neighbour patterns."""

        supported_tids: Bitmap = Bitmap()

        for pattern_candidate in self._generate_allneighbours(pattern):

            # If the pattern_candidate (which is the neighbour of the provided "pattern") has not the same size as
            # the provided "pattern", then it should not be considered as a neighbour. Then skip to the next neighbour
            # candidate.
            if len(pattern_candidate) != len(pattern):
                continue

            # Initialize the patterns transaction ids bitmap with all transactions.
            pattern_tids: Bitmap = Bitmap(range(0, self.n_transactions_))

            # For each item, creates the intersection of the transaction ids. At the end of the loop the transaction ids
            # represents in which transactions the pattern appears.
            # If at one moment the number of transaction is zero, it means the support of the pattern is zero.
            for item in pattern_candidate:
                pattern_tids = pattern_tids.intersection(self.itemid_to_tids_[item])
                if len(pattern_tids) == 0:
                    break

            supported_tids = supported_tids | pattern_tids

            # If the minimum support threshold is met, then the pattern given in parameter is frequent.
            if len(supported_tids) >= min_supp:
                break

        return len(supported_tids) >= min_supp

    def _generate_allneighbours(self, pattern: Pattern) -> Generator[Pattern, None, None]:
        """A generator to generate all the neighbour patterns, including the original one."""
        # If the received pattern is empty, return it.
        if len(pattern) == 0:
            yield frozenset()
        else:
            # Takes one arbitrary item from the pattern.
            current_item: Item = next(iter(pattern))

            # Generate all possible combination of neighbours pattern, including the original one.
            for neighbour_item in {current_item} | self._item_to_neighbours[current_item]:
                for candidate_pattern in self._generate_allneighbours(pattern - frozenset([current_item])):
                    yield frozenset({neighbour_item}) | candidate_pattern


class LCMMax(LCM):
    """
    Linear time Closed item set Miner.

    Adapted to Maximal itemsets (or borders).
    A maximal itemset is an itemset with no frequent superset.

    Parameters
    ----------
    min_supp: int or float, default=0.2
        The minimum support for itemsets to be rendered in the output
        Either an int representing the absolute support, or a float for relative support
        Default to 0.2 (20%)

    max_depth: int, default=20
        Maximum depth for exploration in the search space.
        When going into recursion, we check if the current depth
        is **strictly greater** than `max_depth`.
        If this is the case, we stop.
        This can avoid cumbersome computation.
        A **root node is considered of depth 0**.

    n_jobs : int, default=1
        The number of jobs to use for the computation. Each single item is attributed a job
        to discover potential itemsets, considering this item as a root in the search space.
        **Processes are preffered** over threads.

    See Also
    --------
    LCM
    """

    def _inner(self, p_tids, limit, depth=0):
        if depth >= self.max_depth:
            return
        p, tids = p_tids
        # project and reduce DB w.r.t P
        cp = (
            item
            for item, ids in reversed(self.itemid_to_tids_.items())
            if tids.issubset(ids)
            if item not in p
        )

        max_k = next(
            cp, None
        )  # items are in reverse order, so the first consumed is the max

        if max_k and max_k == limit:
            p_prime = (
                p | set(cp) | {max_k}
            )  # max_k has been consumed when calling next()

            candidates = self.itemid_to_tids_.keys() - p_prime
            candidates = candidates[: candidates.bisect_left(limit)]

            no_cand = True
            for new_limit in candidates:
                ids = self.itemid_to_tids_[new_limit]
                if tids.intersection_len(ids) >= self._min_supp:
                    no_cand = False
                    # get new pattern and its associated tids
                    new_p_tids = (p_prime, tids.intersection(ids))
                    yield from self._inner(new_p_tids, new_limit, depth + 1)

            # only if no child node. This is how we PRE-check for maximality
            if no_cand:
                yield tuple(sorted(p_prime)), tids, depth

    def discover(self, *args, **kwargs):  # pylint: disable=signature-differs
        patterns = super().discover(*args, **kwargs)
        maximums = [tuple(sorted(x)) for x in filter_maximal(patterns["itemset"])]
        return patterns[patterns.itemset.isin(maximums)]

    setattr(discover, "__doc__", LCM.discover.__doc__.replace("closed", "maximal"))
