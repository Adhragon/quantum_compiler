import pytest
from qubit_allocation.layout_generator import all_layouts, number_of_layouts, validate_layout


def test_7p5_count_is_2520():
    assert number_of_layouts(5, 7) == 2520
    assert len(list(all_layouts(5, 7))) == 2520


def test_layouts_are_injective_and_in_range():
    for layout in all_layouts(3, 4):
        assert validate_layout(layout, 3, 4)


def test_invalid_mapping_rejected():
    assert not validate_layout([0, 0, 1], 3, 4)
    assert not validate_layout([0, 1, 4], 3, 4)
    with pytest.raises(ValueError):
        list(all_layouts(5, 4))
