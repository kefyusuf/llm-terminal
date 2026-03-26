import pytest

from release_check_helpers import unpack_search_output


def test_unpack_search_output_accepts_two_tuple_shape():
    results, errors = unpack_search_output(([{"name": "a"}], []))
    assert results == [{"name": "a"}]
    assert errors == []


def test_unpack_search_output_accepts_three_tuple_shape():
    results, errors = unpack_search_output(([{"name": "a"}], ["warn"], True))
    assert results == [{"name": "a"}]
    assert errors == ["warn"]


def test_unpack_search_output_rejects_unexpected_shape():
    with pytest.raises(ValueError, match="unexpected provider search output shape"):
        unpack_search_output(([{"name": "a"}],))
