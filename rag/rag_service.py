"""
总结服务类：用户提问，搜索参考资料，将提问和参考资料提交给模型，让模型总结服务
"""
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from model.factory import chat_model
from rag.vector_store import VectorStoreService
from utils.logger_handler import logger
from utils.prompt_loader import load_rag_summarize_prompt


class RagSummarizerService:
    def __init__(self):
        self.vector_store = VectorStoreService()
        self.prompt_text = load_rag_summarize_prompt()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = chat_model
        self.chain = self._init_chain()


    def _init_chain(self):
        chain = self.prompt_template | self.model | StrOutputParser()
        return chain

    def retriever_docs(self,query:str) -> list[Document]:
        return self.vector_store.search(query)


    def rag_summarize(self,query:str) -> str:
        context_docs = self.retriever_docs(query)
        #记录检索到了什么 —— 排查"答非所问"时，第一件事就是看它检索回了哪些资料。
        #走日志而不是 print：print 会把整段提示词刷到服务端控制台。
        sources = [d.metadata.get("source", "?").split("\\")[-1] for d in context_docs]
        logger.info(f"[rag]检索「{query}」→ {len(context_docs)} 条，来源：{sources}")
        context = ""
        counter = 0
        for doc in context_docs:
            counter += 1
            context += f"【参考资料{counter}】:参考资料：{doc.page_content} | 参考元数据{doc.metadata}\n"

        return self.chain.invoke(
            {
                "input": query,
                "context": context,
            }
        )


if __name__ == "__main__":
    rag = RagSummarizerService()
    print(rag.rag_summarize("中户型适合哪些机器人"))


