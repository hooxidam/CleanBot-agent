"""业务时间来源。

模拟厂家数据使用固定业务时钟，确保设备状态、故障历史和保修计算
处于同一时间线；删除 ``simulation_now`` 配置后自动回退到真实时间。
"""
from datetime import datetime

from utils.config_handler import agent_conf


def business_now() -> datetime:
    """返回当前业务时间；模拟环境优先使用配置中的固定时间。"""
    raw_value = agent_conf.get("simulation_now")
    if not raw_value:
        return datetime.now()

    try:
        return datetime.fromisoformat(str(raw_value))
    except ValueError as exc:
        raise ValueError(
            "config/agent.yml 中 simulation_now 必须是 ISO 时间，"
            "例如 2026-07-17 12:00:00"
        ) from exc

