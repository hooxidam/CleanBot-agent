"""
用户档案与设备使用数据。

数据源 data/external/records.csv 在真实产品中对应两部分：
  - 用户档案：user_id、家居环境特征（真实系统里来自账号/用户资料）；
  - 设备遥测：每月上报的清洁效率、耗材状态等（真实系统里来自设备联网上报的数据库）。
本项目用一个 CSV 同时模拟两者。上层只通过本模块的函数访问，
将来换成真实的账号系统 + 遥测 API 时，只需改本模块，Agent 侧无需改动。

CSV 列：用户ID, 特征, 清洁效率, 耗材, 对比, 时间(YYYY-MM)
"""
import csv

from utils.config_handler import agent_conf
from utils.logger_handler import logger
from utils.path_tool import get_abs_path

# {user_id: {"feature": str, "records": {月份: {清洁效率/耗材/对比}}}}
_data: dict[str, dict] = {}


def _load() -> dict[str, dict]:
    """解析 CSV，结果缓存。只在第一次访问时读文件。"""
    if _data:
        return _data

    path = get_abs_path(agent_conf["external_data_path"])
    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            #用标准库 csv 解析：字段带引号，csv 模块能正确处理，
            #比 split(",") 稳妥(字段内若出现逗号也不会切错)
            for row in csv.DictReader(f):
                user_id = (row.get("用户ID") or "").strip()
                month = (row.get("时间") or "").strip()
                if not user_id or not month:
                    continue

                entry = _data.setdefault(user_id, {"feature": "", "records": {}})
                #特征属于用户档案，每月一行是冗余数据，取一次即可
                if not entry["feature"]:
                    entry["feature"] = (row.get("特征") or "").strip()
                entry["records"][month] = {
                    "清洁效率": (row.get("清洁效率") or "").strip(),
                    "耗材": (row.get("耗材") or "").strip(),
                    "对比": (row.get("对比") or "").strip(),
                }
    except FileNotFoundError:
        logger.error(f"[user_store]数据文件不存在：{path}")
        return {}
    except Exception as e:
        logger.error(f"[user_store]读取数据失败：{e}", exc_info=True)
        return {}

    logger.info(f"[user_store]加载完成：{len(_data)} 个用户")
    return _data


def list_users() -> list[dict]:
    """全部用户及其家居特征：[{"user_id": "1001", "feature": "65㎡公寓 | 单身 | 木地板"}, ...]"""
    return [{"user_id": uid, "feature": entry["feature"]}
            for uid, entry in sorted(_load().items())]


def user_exists(user_id: str) -> bool:
    return user_id in _load()


def get_latest_month(user_id: str) -> str | None:
    """该用户最近有数据的月份。月份是 YYYY-MM，字符串排序即时间排序。"""
    entry = _load().get(user_id)
    if not entry or not entry["records"]:
        return None
    return max(entry["records"])


def get_record(user_id: str, month: str) -> dict | None:
    """某用户某月的使用记录"""
    entry = _load().get(user_id)
    return entry["records"].get(month) if entry else None


def get_latest_snapshot(user_id: str) -> dict | None:
    """该用户最近一个月的设备数据摘要（耗材/清洁表现），用于注入对话上下文。

    注意：家居环境类信息（养宠/地面/面积）已由 store.profile_store 负责，
    那部分需要区分"数据从哪来"，不适合和设备遥测混在一起。
    """
    entry = _load().get(user_id)
    if not entry:
        return None

    latest_month = get_latest_month(user_id)
    latest = entry["records"].get(latest_month, {}) if latest_month else {}
    return {
        "user_id": user_id,
        "latest_month": latest_month,
        "consumables": latest.get("耗材", ""),
        "efficiency": latest.get("清洁效率", ""),
    }
