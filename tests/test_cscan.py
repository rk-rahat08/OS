from src.schedulers import cscan


def test_cscan_up_counts_wraparound():
    result = cscan([98, 183, 37, 122, 14, 124, 65, 67], 53, 0, 199, "up")
    assert result.service_order == [65, 67, 98, 122, 124, 183, 14, 37]
    assert result.total_head_movement == 382
    assert result.movement_sequence.count(199) == 1
    assert 0 in result.movement_sequence


def test_cscan_down_counts_wraparound():
    result = cscan([10, 20, 150, 180], 100, 0, 199, "down")
    assert result.service_order == [20, 10, 180, 150]
    assert result.total_head_movement == 348


def test_cscan_no_wrap_when_nothing_on_other_side():
    result = cscan([120, 150], 100, 0, 199, "up")
    assert result.total_head_movement == 99
