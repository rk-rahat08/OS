from src.schedulers import fcfs


def test_fcfs_textbook_movement():
    result = fcfs([98, 183, 37, 122, 14, 124, 65, 67], 53)
    assert result.service_order == [98, 183, 37, 122, 14, 124, 65, 67]
    assert result.total_head_movement == 640


def test_fcfs_empty_and_repeated_requests():
    assert fcfs([], 50).total_head_movement == 0
    result = fcfs([50, 50, 60], 50)
    assert result.total_head_movement == 10
    assert result.service_order == [50, 50, 60]
