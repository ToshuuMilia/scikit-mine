import pandas as pd
import pytest
import numpy as np

from ..lcm import LCM
from ..lcm import LCMMax

# Tests numerals

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
    lcm = LCM(min_supp=3)
    lcm.fit(D)

    for itemid in lcm.itemid_to_tids_.keys():
        item = lcm.itemid_to_item[itemid]
        assert set(lcm.itemid_to_tids_[itemid]) == true_item_to_tids[item]


def test_first_parent_limit_1():
    lcm = LCM(min_supp=3)
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
    lcm = LCM(min_supp=3)
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
    lcm = LCM(min_supp=3)
    lcm.fit(D)

    limit = 3
    limit_id = lcm.item_to_itemid[limit]
    tids = lcm.itemid_to_tids_[limit_id]
    itemset, tids, _ = next(lcm._inner((frozenset(), tids), limit_id), NULL_RESULT)
    assert itemset == (limit,)
    assert len(tids) == 3


def test_first_parent_limit_4():
    lcm = LCM(min_supp=3)
    lcm.fit(D)

    limit = 4
    limit_id = lcm.item_to_itemid[limit]
    tids = lcm.itemid_to_tids_[limit_id]
    itemset, tids, _ = next(lcm._inner((frozenset(), tids), limit_id), NULL_RESULT)
    assert itemset == (limit,)
    assert len(tids) == 5


def test_first_parent_limit_5():
    lcm = LCM(min_supp=3)
    lcm.fit(D)

    limit = 5
    limit_id = lcm.item_to_itemid[limit]
    tids = lcm.itemid_to_tids_[limit_id]
    itemset, tids, _ = next(lcm._inner((frozenset(), tids), limit_id), NULL_RESULT)
    assert itemset == (2, 5)
    assert len(tids) == 4


def test_first_parent_limit_6():
    lcm = LCM(min_supp=3)
    lcm.fit(D)

    limit = 6
    limit_id = lcm.item_to_itemid[limit]
    tids = lcm.itemid_to_tids_[limit_id]
    itemset, tids, _ = next(lcm._inner((frozenset(), tids), limit_id), NULL_RESULT)
    assert itemset == (4, 6)
    assert len(tids) == 4


def test_lcm_empty_fit():
    # 1. test with a min_supp high above the maximum supp
    lcm = LCM(min_supp=100)
    res = lcm.fit_discover(D)
    assert isinstance(res, pd.DataFrame)
    assert res.empty, f"{res}"

    # 2. test with empty data
    lcm = LCM(min_supp=3)
    res = lcm.fit_discover([])
    assert isinstance(res, pd.DataFrame)
    assert res.empty


def test_lcm_discover():
    lcm = LCM(min_supp=3)
    patterns = lcm.fit_discover(D)  # get new pattern set

    for itemset in patterns.itemset:
        assert itemset in list(true_patterns.itemset)

    for true_itemset in true_patterns.itemset:
        assert true_itemset in list(patterns.itemset)

    pd.testing.assert_series_equal(
        patterns.support, true_patterns.support, check_dtype=False
    )


def test_lcm_discover_max_depth():
    patterns = LCM(min_supp=3, max_depth=1).fit_discover(D, return_depth=True)
    assert (patterns.depth < 2).all()


def test_relative_support():
    lcm = LCM(min_supp=0.4)  # 40% out of 7 transactions ~= 3
    lcm.fit(D)
    np.testing.assert_almost_equal(lcm._min_supp, 2.8, 2)

    for item_id in lcm.itemid_to_tids_.keys():
        assert set(lcm.itemid_to_tids_[item_id]) == true_item_to_tids[lcm.itemid_to_item[item_id]]


def test_lcm_max():
    lcm = LCMMax(min_supp=3)
    patterns = lcm.fit_discover(D)
    assert set(patterns.itemset) == {
        (1, 4, 6),
        (2, 5),
        (2, 4),
        (3,),
    }

#####################
# Test alphabetical #
#####################

D_a = [
    ["chocolat", "lait", "biscuit", "camembert", "plaid", "café"],
    ["lait", "biscuit", "plaid"],
    ["lait", "plaid"],
    ["chocolat", "lait", "camembert", "plaid", "café"],
    ["lait", "camembert"],
    ["chocolat", "camembert", "café"],
    ["biscuit", "camembert", "café"],
]

true_item_to_tids_a = {
    "chocolat": {0, 3, 5},
    "lait": {0, 1, 2, 3, 4},
    "biscuit": {0, 1, 6},
    "camembert": {0, 3, 4, 5, 6},
    "plaid": {0, 1, 2, 3},
    "café": {0, 3, 5, 6},
}

true_patterns_a = pd.DataFrame(
    [  # from D with min_supp=3
        [{"camembert"}, 5],
        [{"lait"}, 5],
        [{"camembert", "lait"}, 3],
        [{"lait", "plaid"}, 4],
        [{"café", "camembert"}, 4],
        [{"café", "camembert", "chocolat"}, 3],
        [{"biscuit"}, 3],
    ],
    columns=["itemset", "support"],
)

true_patterns_a.loc[:, "itemset"] = true_patterns_a.itemset.map(tuple)


def test_lcm_fit_a():
    lcm = LCM(min_supp=3)
    lcm.fit(D_a)

    for item in lcm.itemid_to_tids_.keys():
        assert set(lcm.itemid_to_tids_[item]) == true_item_to_tids_a[item]


def test_first_parent_limit_1_a():
    lcm = LCM(min_supp=3)
    lcm.fit(D_a)

    limit = "chocolat"
    tids = lcm.itemid_to_tids_[limit]

    ## pattern = {"camembert", "café"} -> first parent OK
    itemset, tids, _ = next(lcm._inner((frozenset(["camembert", "café"]), tids), limit), NULL_RESULT)
    assert itemset == ("café", "camembert", "chocolat")
    assert len(tids) == 3

    # pattern = {} -> first parent fails
    itemset, tids, _ = next(lcm._inner((frozenset(), tids), limit), NULL_RESULT)
    assert itemset == None


def test_first_parent_limit_2_a():
    lcm = LCM(min_supp=3)
    lcm.fit(D_a)

    # pattern = {} -> first parent OK
    tids = lcm.itemid_to_tids_["lait"]
    itemset, tids, _ = next(lcm._inner((frozenset(), tids), "lait"), NULL_RESULT)
    assert itemset == ("lait",)
    assert len(tids) == 5

    # pattern = {"camembert"} -> first parent OK
    tids = lcm.itemid_to_tids_["lait"] & lcm.itemid_to_tids_["camembert"]
    itemset, tids, _ = next(lcm._inner((frozenset(["camembert"]), tids), "lait"), NULL_RESULT)
    assert itemset == ("lait", "camembert")
    assert len(tids) == 3


def test_first_parent_limit_3_a():
    lcm = LCM(min_supp=3)
    lcm.fit(D_a)

    tids = lcm.itemid_to_tids_["biscuit"]
    itemset, tids, _ = next(lcm._inner((frozenset(), tids), "biscuit"), NULL_RESULT)
    assert itemset == ("biscuit",)
    assert len(tids) == 3


def test_first_parent_limit_4_a():
    lcm = LCM(min_supp=3)
    lcm.fit(D_a)

    tids = lcm.itemid_to_tids_["camembert"]
    itemset, tids, _ = next(lcm._inner((frozenset(), tids), "camembert"), NULL_RESULT)
    assert itemset == ("camembert",)
    assert len(tids) == 5


def test_first_parent_limit_5_a():
    lcm = LCM(min_supp=3)
    lcm.fit(D_a)

    tids = lcm.itemid_to_tids_["plaid"]
    itemset, tids, _ = next(lcm._inner((frozenset(), tids), "plaid"), NULL_RESULT)
    assert itemset == ("lait", "plaid")
    assert len(tids) == 4


def test_first_parent_limit_6_a():
    lcm = LCM(min_supp=3)
    lcm.fit(D_a)

    tids = lcm.itemid_to_tids_["café"]
    itemset, tids, _ = next(lcm._inner((frozenset(), tids), "café"), NULL_RESULT)
    assert itemset == ("camembert", "café")
    assert len(tids) == 4


def test_lcm_empty_fit_a():
    # 1. test with a min_supp high above the maximum supp
    lcm = LCM(min_supp=100)
    res = lcm.fit_discover(D_a)
    assert isinstance(res, pd.DataFrame)
    assert res.empty

    # 2. test with empty data
    lcm = LCM(min_supp=3)
    res = lcm.fit_discover([])
    assert isinstance(res, pd.DataFrame)
    assert res.empty


def test_lcm_discover_a():
    lcm = LCM(min_supp=3)
    patterns = lcm.fit_discover(D_a)  # get new pattern set

    for itemset, true_itemset in zip(patterns.itemset, true_patterns_a.itemset):
        assert itemset == true_itemset

    pd.testing.assert_series_equal(
        patterns.support, true_patterns_a.support, check_dtype=False
    )


def test_lcm_discover_max_depth_a():
    patterns = LCM(min_supp=3, max_depth=1).fit_discover(D_a, return_depth=True)
    assert not (("chocolat", "camembert", "café") in patterns.itemset.tolist())
    assert (patterns.depth < 2).all()


def test_relative_support_a():
    lcm = LCM(min_supp=0.4)  # 40% out of 7 transactions ~= 3
    lcm.fit(D_a)
    np.testing.assert_almost_equal(lcm._min_supp, 2.8, 2)

    for item in lcm.itemid_to_tids_.keys():
        assert set(lcm.itemid_to_tids_[item]) == true_item_to_tids_a[item]
