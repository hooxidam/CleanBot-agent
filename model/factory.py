import os
from abc import ABC, abstractmethod
from typing import Optional
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import (
    DashScopeEmbeddings,
    HuggingFaceEmbeddings,
)
from utils.config_handler import rag_conf

#从环境变量读取密钥(由 config_handler 的 load_dotenv 从 .env 注入)，避免密钥硬编码进代码/配置
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    raise RuntimeError(
        "未检测到环境变量 DASHSCOPE_API_KEY。请在项目根目录的 .env 文件中配置："
        "DASHSCOPE_API_KEY=你的密钥（可参考 .env.example）"
    )

#通义千问的 OpenAI 兼容接口地址。
#之所以用 ChatOpenAI 而不是 ChatTongyi：ChatTongyi 在 streaming=True 时
#拼接工具调用参数会产生非法 JSON，导致 DashScope 报 InvalidParameter；
#OpenAI 兼容协议对"流式 + 工具调用"处理标准可靠，两者可兼得。
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

#===== 超时与重试 =====
#不配这些的话，DashScope 抽风或网络卡住时请求会一直挂着：
#用户对着闪烁的光标干等，既不知道出了什么事，也没法取消。
#单次模型调用的整体超时。留得比较宽是因为报告生成这类长输出本身就慢；
#注意流式响应是逐块返回的，只要 token 在持续吐就不会触发。
CHAT_TIMEOUT_SECONDS = 90
#流式过程中"两个数据块之间"的最长间隔。整体超时管不住"连上了但卡死不吐字"的情况，
#这个参数才管得住 —— 卡住 30 秒没有新内容就判定为异常，而不是让用户一直等。
CHAT_STREAM_CHUNK_TIMEOUT_SECONDS = 30
#瞬时网络抖动、限流(429)自动重试。重试由 SDK 内部完成，带指数退避。
#不宜过多：模型调用本身就慢，重试太多会让用户等更久。
MAX_RETRIES = 2


class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> Optional[Embeddings|BaseChatModel]:
        pass


class ChatModelFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings|BaseChatModel]:
        return ChatOpenAI(
            model=rag_conf["chat_model_name"],
            api_key=DASHSCOPE_API_KEY,
            base_url=DASHSCOPE_BASE_URL,
            streaming=True,   #开启流式：模型才会逐 token 返回，配合 SSE 实现"吐字"效果
            request_timeout=CHAT_TIMEOUT_SECONDS,
            stream_chunk_timeout=CHAT_STREAM_CHUNK_TIMEOUT_SECONDS,
            max_retries=MAX_RETRIES,
        )

class EmbeddingFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        provider = rag_conf.get("embedding_provider", "local").lower()

        if provider == "local":
            return HuggingFaceEmbeddings(
                model_name=rag_conf["embedding_model_name"],
                model_kwargs={
                    "device": rag_conf.get("embedding_device", "cpu"),
                },
                encode_kwargs={
                    "normalize_embeddings": rag_conf.get(
                        "embedding_normalize",
                        True,
                    ),
                },
            )

        if provider == "dashscope":
            return DashScopeEmbeddings(
                model=rag_conf["embedding_model_name"],
                dashscope_api_key=DASHSCOPE_API_KEY,
                max_retries=MAX_RETRIES,
            )

        raise ValueError(
            f"不支持的 embedding_provider: {provider}"
        )


chat_model = ChatModelFactory().generator()
embedding_model = EmbeddingFactory().generator()
