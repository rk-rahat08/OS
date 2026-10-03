from src.schedulers import run_all


def test_every_scheduler_services_requests_exactly_once():
    requests = [0, 50, 50, 100, 199, 20]
    for result in run_all(requests, 100).values():
        assert sorted(result.service_order) == sorted(requests)
        assert len(result.service_order) == len(requests)
        assert result.total_head_movement >= 0
