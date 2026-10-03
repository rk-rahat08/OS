from src.schedulers import scan


def test_scan_up_reaches_boundary_then_reverses():
    result = scan([98, 183, 37, 122, 14, 124, 65, 67], 53, 0, 199, "up")
    assert result.service_order == [65, 67, 98, 122, 124, 183, 37, 14]
    assert result.total_head_movement == 331
    assert 199 in result.movement_sequence


def test_scan_down_and_boundaries():
    result = scan([0, 10, 150, 199], 100, 0, 199, "down")
    assert result.service_order == [10, 0, 150, 199]
    assert result.total_head_movement == 299


def test_scan_empty_and_all_above():
    assert scan([], 100).movement_sequence == [100]
    result = scan([120, 150], 100)
    assert result.total_head_movement == 99
    assert result.service_order == [120, 150]
