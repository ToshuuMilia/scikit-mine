import pandas as pd
import pytest
import numpy as np

from ..lcm import LCMNeighbours


D = [
    [1, 2, 3, 4, 5, 6],
    [2, 3, 5],
    [2, 5],
    [1, 2, 4, 5, 6],
    [2, 4],
    [1, 4, 6],
    [3, 4, 6],
]

true_item_to_tids = {
    1: {0, 3, 5},
    2: {0, 1, 2, 3, 4},
    3: {0, 1, 6},
    4: {0, 3, 4, 5, 6},
    5: {0, 1, 2, 3},
    6: {0, 3, 5, 6},
}

true_patterns = pd.DataFrame(
    [  # from D with min_supp=3
        [{2}, 5],
        [{4}, 5],
        [{2, 4}, 3],
        [{2, 5}, 4],
        [{4, 6}, 4],
        [{1, 4, 6}, 3],
        [{3}, 3],
    ],
    columns=["itemset", "support"],
)

true_patterns.loc[:, "itemset"] = true_patterns.itemset.map(tuple)

NULL_RESULT = (None, None, 0)


def test_lcm_fit():
    lcm = LCMNeighbours(min_supp=3)
    lcm.fit(D)

    for itemid in lcm.itemid_to_tids_.keys():
        item = lcm.itemid_to_item[itemid]
        assert set(lcm.itemid_to_tids_[itemid]) == true_item_to_tids[item]


def test_first_parent_limit_1():
    lcm = LCMNeighbours(min_supp=3)
    lcm.fit(D)

    limit = 1
    limit_id = lcm.item_to_itemid[limit]
    tids = lcm.itemid_to_tids_[limit_id]

    ## pattern = {4, 6} -> first parent OK
    itemset, tids, _ = next(lcm._inner((frozenset([lcm.item_to_itemid[4], lcm.item_to_itemid[6]]), tids), limit_id), NULL_RESULT)
    assert itemset == (1, 4, 6)
    assert len(tids) == 3

    # pattern = {} -> first parent fails
    itemset, tids, _ = next(lcm._inner((frozenset(), tids), limit_id), NULL_RESULT)
    assert itemset == (1, 4, 6)


def test_first_parent_limit_2():
    lcm = LCMNeighbours(min_supp=3)
    lcm.fit(D)

    # pattern = {} -> first parent OK
    limit = 2
    limit_id = lcm.item_to_itemid[limit]
    tids = lcm.itemid_to_tids_[limit_id]
    itemset, tids, _ = next(lcm._inner((frozenset(), tids), limit_id), NULL_RESULT)
    assert itemset == (limit,)
    assert len(tids) == 5

    # pattern = {4} -> first parent OK
    four = 4
    four_id = lcm.item_to_itemid[four]
    tids = lcm.itemid_to_tids_[limit_id] & lcm.itemid_to_tids_[four_id]
    itemset, tids, _ = next(lcm._inner((frozenset([four_id]), tids), limit_id), NULL_RESULT)
    assert itemset == (limit, four)
    assert len(tids) == 3


def test_first_parent_limit_3():
    lcm = LCMNeighbours(min_supp=3)
    lcm.fit(D)

    limit = 3
    limit_id = lcm.item_to_itemid[limit]
    tids = lcm.itemid_to_tids_[limit_id]
    itemset, tids, _ = next(lcm._inner((frozenset(), tids), limit_id), NULL_RESULT)
    assert itemset == (limit,)
    assert len(tids) == 3


def test_first_parent_limit_4():
    lcm = LCMNeighbours(min_supp=3)
    lcm.fit(D)

    limit = 4
    limit_id = lcm.item_to_itemid[limit]
    tids = lcm.itemid_to_tids_[limit_id]
    itemset, tids, _ = next(lcm._inner((frozenset(), tids), limit_id), NULL_RESULT)
    assert itemset == (limit,)
    assert len(tids) == 5


def test_first_parent_limit_5():
    lcm = LCMNeighbours(min_supp=3)
    lcm.fit(D)

    limit = 5
    limit_id = lcm.item_to_itemid[limit]
    tids = lcm.itemid_to_tids_[limit_id]
    itemset, tids, _ = next(lcm._inner((frozenset(), tids), limit_id), NULL_RESULT)
    assert itemset == (2, 5)
    assert len(tids) == 4


def test_first_parent_limit_6():
    lcm = LCMNeighbours(min_supp=3)
    lcm.fit(D)

    limit = 6
    limit_id = lcm.item_to_itemid[limit]
    tids = lcm.itemid_to_tids_[limit_id]
    itemset, tids, _ = next(lcm._inner((frozenset(), tids), limit_id), NULL_RESULT)
    assert itemset == (4, 6)
    assert len(tids) == 4


def test_lcm_empty_fit():
    # 1. test with a min_supp high above the maximum supp
    lcm = LCMNeighbours(min_supp=100)
    res = lcm.fit_discover(D)
    assert isinstance(res, pd.DataFrame)
    assert res.empty, f"{res}"

    # 2. test with empty data
    lcm = LCMNeighbours(min_supp=3)
    res = lcm.fit_discover([])
    assert isinstance(res, pd.DataFrame)
    assert res.empty


def test_lcm_discover():
    lcm = LCMNeighbours(min_supp=3)
    patterns = lcm.fit_discover(D)  # get new pattern set

    for itemset in patterns.itemset:
        assert itemset in list(true_patterns.itemset)

    for true_itemset in true_patterns.itemset:
        assert true_itemset in list(patterns.itemset)

    pd.testing.assert_series_equal(
        patterns.support, true_patterns.support, check_dtype=False
    )


def test_lcm_discover_max_depth():
    patterns = LCMNeighbours(min_supp=3, max_depth=1).fit_discover(D, return_depth=True)
    assert (patterns.depth < 2).all()


def test_relative_support():
    lcm = LCMNeighbours(min_supp=0.4)  # 40% out of 7 transactions ~= 3
    lcm.fit(D)
    np.testing.assert_almost_equal(lcm._min_supp, 2.8, 2)

    for item_id in lcm.itemid_to_tids_.keys():
        assert set(lcm.itemid_to_tids_[item_id]) == true_item_to_tids[lcm.itemid_to_item[item_id]]


D2 = [
    ["a", "b", "c", "d", "g"],
    ["a", "c", "d", "h"],
    ["a", "b", "e", "f", "j", "l"],
    ["a", "c", "d"],
    ["b", "f", "i", "k"],
    ["a", "c", "e", "h"],
    ["b", "c", "d", "g", "i"],
    ["a", "c", "f", "j", "k"],
    ["e", "g", "i", "k", "l"],
    ["b", "c", "d", "e", "f"]
]

item_to_neighbours2 = {
    "a": ["c", "d", "l"],
    "b": list(),
    "c": ["a", "d"],
    "d": ["a", "c", "g", "e"],
    "e": ["a", "c", "d", "l"],
    "f": ["h", "i", "k"],
    "g": ["d"],
    "h": ["f", "i", "k"],
    "i": ["f", "h"],
    "j": list(),
    "k": ["f", "h"],
    "l": ["a", "e"]
}

true_item_to_tids2 = {
    1: {0, 3, 5},
    2: {0, 1, 2, 3, 4},
    3: {0, 1, 6},
    4: {0, 3, 4, 5, 6},
    5: {0, 1, 2, 3},
    6: {0, 3, 5, 6},
}

true_patterns2 = pd.DataFrame(
    [  # from D with min_supp=3
        [{2}, 5],
        [{4}, 5],
        [{2, 4}, 3],
        [{2, 5}, 4],
        [{4, 6}, 4],
        [{1, 4, 6}, 3],
        [{3}, 3],
    ],
    columns=["itemset", "support"],
)

true_patterns2.loc[:, "itemset"] = true_patterns2.itemset.map(tuple)


def test_lcm_neighboursgen():
    lcm = LCMNeighbours(min_supp=5)
    lcm.fit(D2, item_to_neighbours=item_to_neighbours2)
    pattern = frozenset([lcm.item_to_itemid["l"]])

    neighbours = list(lcm._generate_allneighbours(pattern))

    assert frozenset([lcm.item_to_itemid["l"]]) in neighbours
    for neighbour in item_to_neighbours2["l"]:
        assert frozenset([lcm.item_to_itemid[neighbour]]) in neighbours
