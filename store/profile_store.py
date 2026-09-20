"""
用户档案：家居环境信息。

## 为什么要区分"数据从哪来"

档案里的信息可靠性差很远，混在一起就会自欺欺人：

| 字段 | 真实来源 | 可靠性 |
|------|---------|--------|
| 房屋面积 | **设备自己测的**（SLAM 建图算出清扫面积），自动上报 | 高，用户想瞒也瞒不住 |
| 是否有地毯 | 中高端机型有地毯识别（超声波/压力传感器） | 中高 |
| 地面材质 | 多数机型分不出木地板/瓷砖，只能靠**用户在 App 标注** | 中，依赖用户配合 |
| 是否养宠 | 开过"宠物模式"、AI 视觉识别、或**用户主动说** | 中 |
| 家庭成员 | **设备根本测不出来**，只能用户自己讲 | 低，且涉及隐私，不应主动打听 |

所以每个字段都带 source：
  - device  设备测得，可信
  - user    用户告知，可信
  - unknown **不知道** —— 这个状态必须显式存在，
            Agent 才能意识到"我不知道"，进而在需要时主动询问，而不是默默假设。

## 种子数据的来历

records.csv 的「特征」字段形如 "65㎡公寓 | 单身 | 木地板"，
中间那段有时是家庭成员（单身/老人/儿童），有时是宠物（1狗/2猫）——
知道 1003 养狗的，就不知道他家几口人；知道 1001 单身的，就不知道他养不养宠。
这里如实地把残缺保留下来，而不是编一个默认值糊上去。
"""
from datetime import datetime

from store.db import connect
from store.user_store import list_users
from utils.logger_handler import logger

#档案字段 -> 中文名，供拼提示词与日志
FIELD_LABELS = {
    "home_size": "房屋面积/户型",
    "floor_type": "地面材质",
    "has_pet": "是否养宠物",
    "has_carpet": "是否有地毯",
    "household": "家庭成员情况",
}
#哪些字段该由设备自动测得（这些不该去问用户）
DEVICE_FIELDS = {"home_size", "has_carpet"}
#哪些字段值得在需要时主动询问（家庭成员涉及隐私，不主动打听）
ASKABLE_FIELDS = {"floor_type", "has_pet"}

_PET_HINTS = ("狗", "猫", "宠")
_CARPET_HINTS = ("地毯",)


def _setup():
    with connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id    TEXT PRIMARY KEY,
                home_size  TEXT, home_size_src  TEXT,
                floor_type TEXT, floor_type_src TEXT,
                has_pet    TEXT, has_pet_src    TEXT,
                has_carpet TEXT, has_carpet_src TEXT,
                household  TEXT, household_src  TEXT,
                updated_at TEXT
            )
        """)


def _parse_feature(feature: str) -> dict:
    """把 "65㎡公寓 | 单身 | 木地板" 拆成结构化字段。
    中间那段可能是宠物也可能是家庭成员，按关键词判断 —— 判断不出来的就留空(unknown)。"""
    parts = [p.strip() for p in (feature or "").split("|")]
    size = parts[0] if len(parts) > 0 else ""
    middle = parts[1] if len(parts) > 1 else ""
    floor = parts[2] if len(parts) > 2 else ""

    is_pet = any(h in middle for h in _PET_HINTS)
    return {
        #面积来自设备建图，是真实可信的
        "home_size": (size, "device") if size else (None, "unknown"),
        #地面材质：现实中多数机型分不出，算用户在 App 标注过
        "floor_type": (floor, "user") if floor else (None, "unknown"),
        #中间段是宠物才记宠物，否则养宠状态就是"不知道"
        "has_pet": (middle, "user") if is_pet else (None, "unknown"),
        #地毯：设备有识别能力，从地面材质里能判断
        "has_carpet": ("是", "device") if any(h in floor for h in _CARPET_HINTS)
                      else (("否", "device") if floor else (None, "unknown")),
        #家庭成员：中间段不是宠物时才是它
        "household": (middle, "user") if (middle and not is_pet) else (None, "unknown"),
    }


def _seed_if_empty():
    """首次运行时，用 records.csv 里已有的信息初始化档案。
    真实系统里这些信息本就散落在用户资料、设备识别记录里，这里模拟"已经掌握的部分"。"""
    with connect() as conn:
        if conn.execute("SELECT COUNT(*) FROM user_profiles").fetchone()[0] > 0:
            return
        rows = []
        for u in list_users():
            f = _parse_feature(u["feature"])
            rows.append((
                u["user_id"],
                f["home_size"][0], f["home_size"][1],
                f["floor_type"][0], f["floor_type"][1],
                f["has_pet"][0], f["has_pet"][1],
                f["has_carpet"][0], f["has_carpet"][1],
                f["household"][0], f["household"][1],
                datetime.now().isoformat(timespec="seconds"),
            ))
        conn.executemany(
            "INSERT INTO user_profiles VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows)
        logger.info(f"[profile_store]初始化 {len(rows)} 份用户档案")


def get_profile(user_id: str) -> dict | None:
    """返回结构化档案：{字段: {"value":…, "source": device/user/unknown}}"""
    _setup()
    _seed_if_empty()
    row = connect().execute(
        "SELECT * FROM user_profiles WHERE user_id = ?", (user_id,)).fetchone()
    if not row:
        return None

    profile = {"user_id": user_id, "fields": {}}
    for field in FIELD_LABELS:
        profile["fields"][field] = {
            "value": row[field],
            "source": row[f"{field}_src"] or "unknown",
        }
    return profile


def missing_askable_fields(user_id: str) -> list[str]:
    """哪些"值得问用户"的字段还不知道。
    只包含 ASKABLE_FIELDS —— 设备能测的不该问用户，涉及隐私的不主动打听。"""
    profile = get_profile(user_id)
    if not profile:
        return []
    return [f for f in ASKABLE_FIELDS
            if profile["fields"][f]["source"] == "unknown"]


def update_profile(user_id: str, **fields) -> list[str]:
    """记录用户主动告知的档案信息。返回实际更新了哪些字段(中文名)。

    只接受 FIELD_LABELS 里的字段，且一律标记 source=user —— 这是用户自己说的。
    """
    _setup()
    _seed_if_empty()

    valid = {k: v for k, v in fields.items()
             if k in FIELD_LABELS and v not in (None, "")}
    if not valid:
        return []

    sets, params = [], []
    for k, v in valid.items():
        sets.append(f"{k} = ?")
        sets.append(f"{k}_src = ?")
        params.extend([str(v), "user"])
    sets.append("updated_at = ?")
    params.append(datetime.now().isoformat(timespec="seconds"))
    params.append(user_id)

    with connect() as conn:
        conn.execute(f"UPDATE user_profiles SET {', '.join(sets)} WHERE user_id = ?", params)
    updated = [FIELD_LABELS[k] for k in valid]
    logger.info(f"[profile_store]用户 {user_id} 更新档案：{valid}")
    return updated
