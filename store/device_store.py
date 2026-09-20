"""
设备档案、实时状态、故障日志。

对应真实产品中厂家侧的设备数据库/遥测服务：
  - devices.csv       设备档案(型号/购买日/保修) —— 真实系统里来自订单与设备管理系统
  - device_status.csv 实时状态(在线/电量/当前错误) —— 真实系统里来自设备上报的最新心跳
  - error_logs.csv    故障日志 —— 真实系统里来自设备上报的事件流

上层只通过本模块访问。将来对接真实 API 时只改这里，Agent、工具、提示词均无需改动。

⚠️ 三条按"将来要接真实数据"设计的原则：
  1. 所有查询都以 user_id 为入口，由调用方(API 层)从登录态传入，绝不接受模型指定；
  2. 面向模型的查询一律返回**聚合摘要**。真实用户可能有上千条日志，
     原样返回会撑爆上下文、推高成本，还让模型抓不住重点；
  3. **按数据的变化速度决定缓存策略**（见下方）——
     设备状态号称"实时"，就不能一直用进程启动时的那份快照。
"""
import csv
import time
from collections import Counter
from datetime import datetime

from utils.config_handler import agent_conf
from utils.business_clock import business_now
from utils.logger_handler import logger
from utils.path_tool import get_abs_path

#===== 缓存策略：按数据的变化速度区别对待 =====
#设备档案(型号/购买日/保修)几乎不变，可以长期缓存。
#状态和故障日志是持续变化的：真实系统里设备每分钟都在上报，
#若沿用启动时的快照，Agent 会一直汇报几小时前的状态，还管它叫"实时"。
#短 TTL 既保证数据够新，又不至于把下游 API 打爆（现在读的是本地 CSV，
#换成真实 API 后这个策略同样成立）。
STATUS_TTL_SECONDS = 30
ERRORS_TTL_SECONDS = 60

_devices: dict[str, dict] = {}          # user_id -> 设备档案（长期缓存）
_status: dict[str, dict] = {}           # device_id -> 当前状态（短 TTL）
_errors: dict[str, list[dict]] = {}     # device_id -> 故障日志，按时间倒序（短 TTL）
_status_loaded_at: float = 0.0
_errors_loaded_at: float = 0.0


def _read(conf_key: str) -> list[dict]:
    path = get_abs_path(agent_conf[conf_key])
    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    except FileNotFoundError:
        logger.error(f"[device_store]数据文件不存在：{path}")
        return []
    except Exception as e:
        logger.error(f"[device_store]读取 {path} 失败：{e}", exc_info=True)
        return []


def _load_devices():
    """设备档案：几乎不变，加载一次即可"""
    if _devices:
        return
    for row in _read("devices_path"):
        _devices[row["用户ID"]] = row
    logger.info(f"[device_store]加载设备档案：{len(_devices)} 台")


def _load_status():
    """设备状态：过了 TTL 就重新读，保证"实时"名副其实"""
    global _status_loaded_at
    if _status and time.time() - _status_loaded_at < STATUS_TTL_SECONDS:
        return
    _status.clear()
    for row in _read("device_status_path"):
        _status[row["设备ID"]] = row
    _status_loaded_at = time.time()


def _load_errors():
    """故障日志：会持续追加新记录，同样需要按 TTL 刷新"""
    global _errors_loaded_at
    if _errors and time.time() - _errors_loaded_at < ERRORS_TTL_SECONDS:
        return
    _errors.clear()
    for row in _read("error_logs_path"):
        _errors.setdefault(row["设备ID"], []).append(row)
    for logs in _errors.values():
        logs.sort(key=lambda r: r["发生时间"], reverse=True)
    _errors_loaded_at = time.time()


def get_device(user_id: str) -> dict | None:
    """设备档案 + 保修状态(保修是算出来的，不是存的，避免数据过期)"""
    _load_devices()
    row = _devices.get(user_id)
    if not row:
        return None

    warranty_until = row["保修截止"]
    warranty_date = datetime.strptime(warranty_until, "%Y-%m-%d").date()
    remaining_days = (warranty_date - business_now().date()).days
    return {
        "user_id": user_id,
        "device_id": row["设备ID"],
        "model": row["型号"],
        "purchase_date": row["购买日期"],
        "warranty_until": warranty_until,
        # 保修截止日当天仍然属于保修期。
        "in_warranty": remaining_days >= 0,
        "warranty_remaining_days": remaining_days,
        "firmware": row["固件版本"],
    }


def get_status(user_id: str) -> dict | None:
    """设备当前状态(在线/电量/正在做什么/是否报错)。数据按 TTL 刷新，确保"当前"名副其实。"""
    _load_devices()
    _load_status()
    device = _devices.get(user_id)
    if not device:
        return None
    row = _status.get(device["设备ID"])
    if not row:
        return None
    return {
        "device_id": row["设备ID"],
        "model": device["型号"],
        "online": row["在线状态"],
        "battery": row["电量"],
        "state": row["当前状态"],
        "error_code": row["错误码"],
        "error_name": row["错误名称"],
        "updated_at": row["更新时间"],
    }


def summarize_errors(user_id: str, days: int = 7) -> dict | None:
    """最近 N 天的故障**聚合摘要**。

    刻意不返回原始日志列表：真实设备可能几天就上百条事件，
    原样喂给模型既贵又没用。这里按错误类型聚合成"发生了什么、几次、最近一次何时"，
    模型据此就能判断是偶发还是反复出现的老毛病。
    """
    _load_devices()
    _load_errors()
    device = _devices.get(user_id)
    if not device:
        return None

    logs = _errors.get(device["设备ID"], [])
    cutoff = business_now().timestamp() - days * 86400
    recent = [l for l in logs
              if datetime.strptime(l["发生时间"], "%Y-%m-%d %H:%M").timestamp() >= cutoff]

    counts = Counter(f"{l['错误码']}|{l['错误名称']}" for l in recent)
    items = []
    for key, times in counts.most_common():
        code, name = key.split("|")
        latest = next(l for l in recent if l["错误码"] == code)
        items.append({
            "error_code": code,
            "error_name": name,
            "times": times,
            "last_occurred": latest["发生时间"],
            "detail": latest["详情"],
        })

    return {"days": days, "total": len(recent), "items": items}
