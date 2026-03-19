import pytest
import numpy as np

from src.data_preparation.ec_hierarchy_distance import ec_distance, parse_ec_number

def test_parse_ec_number():
    assert parse_ec_number("3.4.21.9") == (3, 4, 21, 9)
    assert parse_ec_number("3.4.21.-") == (3, 4, 21, None)
    assert parse_ec_number("3.-.-.-") == (3, None, None, None)

def test_ec_distance_identical():
    assert ec_distance("3.4.21.9", "3.4.21.9") == 0

def test_ec_distance_level4():
    assert ec_distance("3.4.21.9", "3.4.21.4") == 1

def test_ec_distance_level3():
    assert ec_distance("3.4.21.9", "3.4.24.9") == 2

def test_ec_distance_level2():
    assert ec_distance("3.4.21.9", "3.1.1.1") == 3

def test_ec_distance_level1():
    assert ec_distance("1.1.1.1", "3.4.21.9") == 4

def test_ec_distance_with_wildcards():
    assert ec_distance("3.4.21.-", "3.4.21.9") is None

def test_ec_distance_partial():
    assert ec_distance("3.4.-.-", "3.4.21.9") is None
