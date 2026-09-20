from utils.config_handler import prompts_conf
from utils.path_tool import get_abs_path
from utils.logger_handler import logger


def load_system_prompt():
    try:
        system_prompt_path = get_abs_path(prompts_conf["main_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_system_prompts]配置文件中缺少{e}")
        raise e

    try:
        return open(system_prompt_path,"r",encoding = "utf-8").read()
    except Exception as e:
        logger.error(f"[load_system_prompts]解析系统提示词出错{e}")
        raise e

def load_rag_summarize_prompt():
    try:
        rag_prompt_path = get_abs_path(prompts_conf["rag_summarize_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_rag_summarize_prompts]配置文件中缺少{e}")
        raise e

    try:
        return open(rag_prompt_path,"r",encoding = "utf-8").read()
    except Exception as e:
        logger.error(f"[load_rag_summarize_prompts]解析RAG总结提示词出错{e}")
        raise e


def load_report_prompt():
    try:
        report_prompt_path = get_abs_path(prompts_conf["report_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_report_prompts]配置文件中缺少{e}")
        raise e

    try:
        return open(report_prompt_path,"r",encoding = "utf-8").read()
    except Exception as e:
        logger.error(f"[load_report_prompts]解析报告提示词出错{e}")
        raise e


def load_summary_prompt():
    """加载"对话历史压缩"提示词，供 SummarizationMiddleware 使用"""
    try:
        summary_prompt_path = get_abs_path(prompts_conf["summary_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_summary_prompt]配置文件中缺少{e}")
        raise e

    try:
        return open(summary_prompt_path,"r",encoding = "utf-8").read()
    except Exception as e:
        logger.error(f"[load_summary_prompt]解析摘要提示词出错{e}")
        raise e


if __name__ == "__main__":
    print(load_system_prompt())
    print(load_rag_summarize_prompt())
    print(load_report_prompt())

