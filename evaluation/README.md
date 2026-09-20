# 检索结果评测

本目录只评测知识库检索，不调用聊天模型，也不评价 ReAct 工具选择或最终回答。

## 评测对象

- Embedding：项目配置的本地 `BAAI/bge-small-zh-v1.5`
- 向量库：当前 `chroma_db` 中的 `agent` collection
- 当前生产召回数：Top3
- 对照：Top1、Top5
- 重排序：Chroma Top5 候选经 `BAAI/bge-reranker-base` 排序后保留 Top3

每条人工标签用 `(source_name, chunk_index)` 唯一定位一个文本块。运行前会检查所有标签是否存在于当前索引，避免切分配置变化后继续使用失效标签。

## 运行

在项目根目录执行：

```bash
python -m evaluation.run_retrieval_eval
```

也可以强制对比两条链路：

```bash
python -m evaluation.run_retrieval_eval --reranker off
python -m evaluation.run_retrieval_eval --reranker on
```

报告写入 `evaluation/results/<时间戳>/`，其中：

- `report.md`：汇总指标、通过标准和 Top3 未命中问题。
- `results.json`：所有问题的完整召回结果、逐题指标和耗时。

评测问题由本地 BGE 转换为向量，不会发送给 DashScope，也不会调用 `qwen3.8-max` 生成答案。

人工标注采用 pooled judging：先根据原始文档标注标准答案，再复核系统 Top5 返回的未标注 Chunk；只有确实能够回答问题的 Chunk 才补充为相关结果，避免知识库中的重复答案被错误计为不相关。
