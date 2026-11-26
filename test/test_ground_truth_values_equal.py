import numpy as np

from judge import Judge


def test_numeric_scalars_compare_by_value():
    assert Judge.ground_truth_values_equal(64.0, np.int64(64))


def test_text_and_numeric_scalar_compare_by_numeric_value():
    assert Judge.ground_truth_values_equal("8100000000000.00", 8100000000000.0)
    assert Judge.ground_truth_values_equal(np.int64(10), "10.0")
    assert not Judge.ground_truth_values_equal("not-a-number", 10)


def test_text_and_boolean_scalar_compare_by_boolean_value():
    assert Judge.ground_truth_values_equal("TRUE", np.bool_(True))
    assert Judge.ground_truth_values_equal(False, " false ")
    assert not Judge.ground_truth_values_equal("EURT", True)


def test_two_text_surfaces_remain_exact():
    assert not Judge.ground_truth_values_equal("10.0", "10")
    assert not Judge.ground_truth_values_equal("TRUE", "true")
