import os
import threading

from langchain_core.documents import Document
from langchain_chroma import Chroma
from utils.config_handler import chroma_conf, rag_conf
from model.factory import embedding_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.logger_handler import logger

from utils.file_handler import txt_loader, pdf_loader, get_file_md5_hex, listdir_with_allowed_type
from utils.path_tool import get_abs_path


class VectorStoreService:
    def __init__(self):
        self.vector_store = Chroma(
            collection_name= chroma_conf["collection_name"],
            embedding_function= embedding_model,
            #用绝对路径固定向量库位置，避免因运行目录(CWD)不同而生成多份库
            persist_directory= get_abs_path(chroma_conf["persist_directory"]),
        )
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size= chroma_conf["chunk_size"],
            chunk_overlap= chroma_conf["chunk_overlap"],
            separators= chroma_conf["separators"],
            length_function=len,
        )
        self._reranker = None
        self._reranker_lock = threading.Lock()

    def get_retriever(self):
        return self.vector_store.as_retriever(search_kwargs={"k": chroma_conf["k"]})

    def _get_reranker(self):
        """延迟加载 CrossEncoder，避免服务启动时无条件占用内存。"""
        if self._reranker is not None:
            return self._reranker

        from sentence_transformers import CrossEncoder

        self._reranker = CrossEncoder(
            rag_conf["reranker_model_name"],
            device=rag_conf.get("reranker_device", "cpu"),
        )
        logger.info(
            "[reranker]模型加载完成：%s",
            rag_conf["reranker_model_name"],
        )
        return self._reranker

    def _rerank(self, query: str, documents: list[Document]) -> list[Document]:
        if not documents:
            return []

        with self._reranker_lock:
            reranker = self._get_reranker()
            scores = reranker.predict(
                [(query, document.page_content) for document in documents],
                batch_size=int(rag_conf.get("reranker_batch_size", 8)),
                show_progress_bar=False,
                convert_to_numpy=True,
            )

        ranked: list[tuple[float, Document]] = []
        for score, document in zip(scores, documents):
            numeric_score = float(score)
            document.metadata["rerank_score"] = numeric_score
            ranked.append((numeric_score, document))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [document for _, document in ranked]

    def search(
        self,
        query: str,
        final_k: int | None = None,
        use_reranker: bool | None = None,
        candidate_k: int | None = None,
        strict_reranker: bool = False,
    ) -> list[Document]:
        """向量召回，并按配置选择是否用 CrossEncoder 重排序。"""
        final_k = int(final_k or chroma_conf["k"])
        if final_k <= 0:
            raise ValueError("final_k must be greater than 0")

        if use_reranker is None:
            use_reranker = bool(rag_conf.get("reranker_enabled", False))
        if not use_reranker:
            return self.vector_store.similarity_search(query, k=final_k)

        candidate_k = int(candidate_k or rag_conf.get("reranker_candidate_k", 5))
        candidate_k = max(candidate_k, final_k)
        candidates = self.vector_store.similarity_search(query, k=candidate_k)
        try:
            return self._rerank(query, candidates)[:final_k]
        except Exception:
            logger.exception("[reranker]重排序失败，回退到原始向量排序")
            if strict_reranker:
                raise
            return candidates[:final_k]

    def load_documents(self):
        """
        从数据文件夹内读取文件，转为向量存入向量库
        要计算文件的MD5做去重
        :return:None
        """
        def check_md5_hex(md5_for_check: str) -> bool:
            md5_store_path = get_abs_path(chroma_conf["md5_hex_store"])
            if not os.path.exists(md5_store_path):
                open(md5_store_path, "w", encoding="utf-8").close()
                return False  #md5没处理过

            with open(md5_store_path, "r", encoding="utf-8") as f:
                for line in f.readlines():
                    line = line.strip()
                    if line == md5_for_check:
                        return True  #处理过

            return False   #没处理过

        def save_md5_hex(md5_for_check: str):
            with open(get_abs_path(chroma_conf["md5_hex_store"]), "a", encoding="utf-8") as f:
                f.write(md5_for_check + "\n")

        def get_file_documents(read_path: str):
            if read_path.endswith(".txt"):
                return txt_loader(read_path)

            if read_path.endswith(".pdf"):
                return pdf_loader(read_path)

            return []

        allowed_files_type: list[str] = listdir_with_allowed_type(
            get_abs_path(chroma_conf["data_path"]),
            tuple(chroma_conf["allow_knowledge_file_type"]),
        )

        for path in allowed_files_type:
            #获取文件的md5
            md5_hex = get_file_md5_hex(path)
            if not md5_hex:
                logger.error(f"[加载知识库]{path}的MD5计算失败，跳过")
                continue

            if check_md5_hex(md5_hex):
                logger.info(f"[加载知识库]{path}内容已经存在知识库，跳过")
                continue

            try:
                documnets: list[Document] = get_file_documents(path)
                if not documnets:
                    logger.error(f"[加载知识库]{path}内容为空，跳过")
                    continue

                split_document = self.spliter.split_documents(documnets)

                if not split_document:
                    logger.error(f"[加载知识库]{path}内容分割为空，跳过")
                    continue

                # 为每个文本块写入可复现的业务标识，供检索评测和问题排查使用。
                # Chroma 自带的 UUID 在重建索引后会变化，不能作为评测标签。
                source_name = os.path.basename(path)
                for chunk_index, document in enumerate(split_document):
                    document.metadata.update(
                        {
                            "source_name": source_name,
                            "chunk_index": chunk_index,
                        }
                    )

                #将内容存入向量库
                self.vector_store.add_documents(split_document)

                #记录这个已经处理好的文件的md5，避免下次重复加载
                save_md5_hex(md5_hex)

                logger.info(f"[加载知识库]{path}内容加载完成")
            except Exception as e:
                #exc_info=True 会打印完整的异常信息
                logger.error(f"[加载知识库]{path}内容加载失败，错误信息：{str(e)}",exc_info=True)
                continue


if __name__ == "__main__":
    vs = VectorStoreService()
    vs.load_documents()
    res = vs.search("迷路")
    for r in res:
        print(r.page_content)
        print("_"*20)
