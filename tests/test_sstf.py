from src.schedulers import sstf


def test_sstf_textbook_movement():
    result = sstf([98, 183, 37, 122, 14, 124, 65, 67], 53)
    assert result.service_order == [65, 67, 37, 14, 98, 122, 124, 183]
    assert result.total_head_movement == 236


def test_sstf_lower_cylinder_tie_break():
    result = sstf([40, 60, 30], 50)
    assert result.service_order == [40, 30, 60]


def test_sstf_equal_head_and_single_request():
    assert sstf([100], 100).total_head_movement == 0
    assert sstf([120], 100).total_head_movement == 20
