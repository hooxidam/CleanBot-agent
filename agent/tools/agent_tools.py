"""
Agent 可调用的工具。

两条贯穿全文件的设计原则：

1. user_id 一律从运行时上下文取，绝不做成模型可传的参数。
   如果签名是 fetch(user_id, month)，模型就有权指定查谁 —— 用户一句
   "帮我查下 1005 的记录"，模型很可能真去查了，这就是越权读取他人数据。
   身份来自登录态，模型无权选择，也就没有越权的入口。

2. 面向模型的返回一律是聚合后的摘要，不是原始明细。
   真实用户可能有上千条日志，原样返回会撑爆上下文、推高成本，
   还让模型抓不住重点。
"""
from langgraph.runtime import get_runtime
from langchain_core.tools import tool

from rag.rag_service import RagSummarizerService
from store.device_store import get_device, get_status, summarize_errors
from store.profile_store import update_profile
from store.user_store import get_latest_month, get_record
from utils.logger_handler import logger

rag = RagSummarizerService()


def _current_user_id() -> str | None:
    """当前登录用户，由 API 层在调用 Agent 时注入运行时上下文。
    拿不到就返回 None（例如在命令行里直接跑 Agent，没有登录态）。"""
    try:
        return get_runtime().context.get("user_id")
    except Exception:
        return None


def _flat(text: str) -> str:
    """records.csv 里的 \\n 是字面的反斜杠+n，不是换行符，压成逗号便于模型阅读"""
    return text.replace("\\n", "，").replace("\n", "，")


@tool(description="从知识库中检索扫地/扫拖机器人的专业资料（使用方法、故障处理、维护保养、选购建议等），入参为检索词")
def rag_summarize(query: str) -> str:
    return rag.rag_summarize(query)


@tool(description="查询当前用户的机器人**实时状态**：是否在线、电量、正在做什么、当前是否有故障报错。"
                  "用户反馈机器人异常、不动、报错时应首先调用此工具。无需入参")
def get_device_status() -> str:
    user_id = _current_user_id()
    if not user_id:
        return "无法确定当前用户身份，暂时查询不到设备状态。"

    status = get_status(user_id)
    if not status:
        return "未查询到您名下的设备状态信息。"

    parts = [f"设备 {status['model']}（{status['device_id']}）",
             f"当前{status['online']}",
             f"电量 {status['battery']}%",
             f"状态：{status['state']}"]
    if status["error_code"]:
        parts.append(f"**正在报错：{status['error_code']} {status['error_name']}**")
    else:
        parts.append("当前无故障报错")
    parts.append(f"状态更新于 {status['updated_at']}")
    return "；".join(parts) + "。"


@tool(description="查询当前用户机器人最近一段时间的**故障历史摘要**（按故障类型聚合：发生了什么、共几次、最近一次何时）。"
                  "用于判断某个故障是偶发还是反复出现。入参 days 为回溯天数，默认 7 天")
def get_error_history(days: int = 7) -> str:
    user_id = _current_user_id()
    if not user_id:
        return "无法确定当前用户身份，暂时查询不到故障记录。"

    summary = summarize_errors(user_id, days)
    if not summary:
        return "未查询到您名下的设备信息。"
    if summary["total"] == 0:
        return f"近 {days} 天您的机器人没有故障记录，运行正常。"

    lines = [f"近 {days} 天共发生 {summary['total']} 次故障："]
    for item in summary["items"]:
        lines.append(f"- {item['error_code']} {item['error_name']}：{item['times']} 次，"
                     f"最近一次 {item['last_occurred']}（{item['detail']}）")
    return "\n".join(lines)


@tool(description="查询当前用户的**设备档案**：型号、购买日期、保修状态与剩余保修天数、固件版本。"
                  "用户询问型号功能、是否还在保修期、能否免费维修时调用。无需入参")
def get_device_info() -> str:
    user_id = _current_user_id()
    if not user_id:
        return "无法确定当前用户身份，暂时查询不到设备档案。"

    device = get_device(user_id)
    if not device:
        return "未查询到您名下的设备档案。"

    if device["in_warranty"]:
        warranty = f"在保修期内，保修至 {device['warranty_until']}（剩余 {device['warranty_remaining_days']} 天）"
    else:
        warranty = f"**已过保修期**（保修已于 {device['warranty_until']} 到期，超出 {-device['warranty_remaining_days']} 天）"

    return (f"型号：{device['model']}；设备编号：{device['device_id']}；"
            f"购买日期：{device['purchase_date']}；固件版本：{device['firmware']}；{warranty}。")


@tool(description="查询当前用户某个月的机器人使用记录（清洁效率、耗材状态、同类对比）。"
                  "入参 month 格式为 YYYY-MM，不传则返回最近一个月的记录")
def get_my_usage_record(month: str = None) -> str:
    #身份从上下文取，月份可省略 —— 模型不必先调两个工具去问"我是谁""现在几月"
    user_id = _current_user_id()
    if not user_id:
        return "无法确定当前用户身份，暂时查询不到使用记录。"

    month = month or get_latest_month(user_id)
    if not month:
        return "未查询到您的使用记录。"

    record = get_record(user_id, month)
    if not record:
        latest = get_latest_month(user_id)
        logger.warning(f"[get_my_usage_record]用户 {user_id} 无 {month} 的记录")
        return f"未查询到 {month} 的使用记录。您最近有数据的月份是 {latest}。"

    body = "；".join(f"{k}：{_flat(v)}" for k, v in record.items() if v)
    return f"{month} 使用记录 —— {body}"


@tool(description="记录用户在对话中主动告知的家居环境信息，记录后以后无需再问。"
                  "只填用户**明确说过**的字段，没提到的不要填、更不要猜。"
                  "floor_type=地面材质(如 木地板/瓷砖/地毯)；"
                  "has_pet=养宠情况(如 1只金毛/2只猫/无)；"
                  "has_carpet=是否有地毯(是/否)；household=家庭成员情况(仅当用户主动提及)")
def update_my_profile(floor_type: str = None, has_pet: str = None,
                      has_carpet: str = None, household: str = None) -> str:
    user_id = _current_user_id()
    if not user_id:
        return "无法确定当前用户身份，本次信息未能记录。"

    updated = update_profile(user_id, floor_type=floor_type, has_pet=has_pet,
                             has_carpet=has_carpet, household=household)
    if not updated:
        return "没有需要更新的信息。"
    return f"已记录到您的档案：{'、'.join(updated)}。后续回答会自动结合这些信息，无需重复告知。"


@tool(description="无入参。当且仅当用户明确要求生成/查询个人使用报告时调用，"
                  "调用后系统会自动切换为报告生成模式。非报告场景严禁调用")
def fill_context_for_report():
    return "已切换为报告生成模式"


if __name__ == '__main__':
    print(get_my_usage_record.invoke({}))
