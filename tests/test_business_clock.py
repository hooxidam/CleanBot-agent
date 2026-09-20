import time
from datetime import date, datetime

import pytest

from store import device_store
from utils.business_clock import business_now
from utils.config_handler import agent_conf


def test_business_now_uses_simulation_config(monkeypatch):
    monkeypatch.setitem(agent_conf, "simulation_now", "2026-07-17 12:00:00")

    assert business_now() == datetime(2026, 7, 17, 12, 0, 0)


def test_invalid_simulation_time_fails_fast(monkeypatch):
    monkeypatch.setitem(agent_conf, "simulation_now", "not-a-time")

    with pytest.raises(ValueError, match="simulation_now"):
        business_now()


def test_warranty_uses_business_date(monkeypatch):
    monkeypatch.setitem(agent_conf, "simulation_now", "2026-07-17 12:00:00")
    monkeypatch.setattr(
        device_store,
        "_devices",
        {
            "1004": {
                "用户ID": "1004",
                "设备ID": "demo-device",
                "型号": "智扫 S5",
                "购买日期": "2025-03-05",
                "保修截止": "2027-03-05",
                "固件版本": "5.2.0",
            }
        },
    )

    result = device_store.get_device("1004")

    assert result is not None
    assert result["in_warranty"] is True
    assert result["warranty_remaining_days"] == (
        date(2027, 3, 5) - date(2026, 7, 17)
    ).days


def test_error_window_uses_business_time(monkeypatch):
    monkeypatch.setitem(agent_conf, "simulation_now", "2026-07-17 12:00:00")
    monkeypatch.setattr(
        device_store,
        "_devices",
        {"1004": {"用户ID": "1004", "设备ID": "demo-device"}},
    )
    monkeypatch.setattr(
        device_store,
        "_errors",
        {
            "demo-device": [
                {
                    "设备ID": "demo-device",
                    "错误码": "E5",
                    "错误名称": "主刷缠绕",
                    "发生时间": "2026-07-16 19:39",
                    "详情": "主刷被宠物毛发缠绕",
                },
                {
                    "设备ID": "demo-device",
                    "错误码": "E1",
                    "错误名称": "旧故障",
                    "发生时间": "2026-05-01 10:00",
                    "详情": "已超出30天窗口",
                },
            ]
        },
    )
    monkeypatch.setattr(device_store, "_errors_loaded_at", time.time())

    result = device_store.summarize_errors("1004", days=30)

    assert result is not None
    assert result["total"] == 1
    assert result["items"][0]["error_code"] == "E5"

