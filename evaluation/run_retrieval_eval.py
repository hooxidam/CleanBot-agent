"""Run an offline retrieval-only evaluation against the current Chroma index."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from evaluation.metrics import calculate_metrics, make_chunk_key, mean_metrics
from rag.vector_store import VectorStoreService
from utils.config_handler import rag_conf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = Path(__file__).with_name("retrieval_dataset.jsonl")
DEFAULT_OUTPUT_DIR = Path(__file__).with_name("results")
PASS_THRESHOLDS = {
    "hit_rate_at_k": 0.85,
    "recall_at_k": 0.75,
    "mrr_at_k": 0.75,
    "ndcg_at_k": 0.70,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Chroma retrieval at multiple K values")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument(
        "--reranker",
        choices=["config", "on", "off"],
        default="config",
        help="use configured reranker, force it on, or force vector-only retrieval",
    )
    parser.add_argument("--skip-warmup", action="store_true")
    return parser.parse_args()


def load_dataset(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line:
                continue
            item = json.loads(line)
            item_id = str(item.get("id", "")).strip()
            question = str(item.get("question", "")).strip()
            relevant = item.get("relevant_chunks") or []
            if not item_id or item_id in seen_ids:
                raise ValueError(f"line {line_number}: id is empty or duplicated: {item_id!r}")
            if not question:
                raise ValueError(f"line {line_number}: question must not be empty")
            if not relevant:
                raise ValueError(f"line {line_number}: relevant_chunks must not be empty")
            seen_ids.add(item_id)
            items.append(item)
    if not items:
        raise ValueError("evaluation dataset is empty")
    return items


def available_chunk_keys(vector_store) -> tuple[set[tuple[str, int]], dict[str, int]]:
    records = vector_store.get(include=["metadatas"])
    keys: set[tuple[str, int]] = set()
    sources: Counter[str] = Counter()
    for metadata in records.get("metadatas") or []:
        if not metadata or "source_name" not in metadata or "chunk_index" not in metadata:
            continue
        key = make_chunk_key(metadata["source_name"], metadata["chunk_index"])
        keys.add(key)
        sources[key[0]] += 1
    return keys, dict(sorted(sources.items()))


def validate_labels(items: list[dict[str, Any]], available: set[tuple[str, int]]) -> None:
    missing: list[str] = []
    for item in items:
        for label in item["relevant_chunks"]:
            key = make_chunk_key(label["source_name"], label["chunk_index"])
            if key not in available:
                missing.append(f"{item['id']}: {key}")
    if missing:
        preview = "\n".join(missing[:20])
        raise ValueError(f"dataset contains {len(missing)} missing chunk labels:\n{preview}")


def percentile(values: list[float], percent: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(percent * len(ordered)) - 1)
    return ordered[index]


def evaluate(
    service: VectorStoreService,
    items: list[dict[str, Any]],
    ks: list[int],
    warmup: bool,
    use_reranker: bool,
) -> dict[str, Any]:
    max_k = max(ks)
    if warmup:
        service.search(
            items[0]["question"],
            final_k=max_k,
            use_reranker=use_reranker,
            strict_reranker=use_reranker,
        )

    metric_buckets = {k: [] for k in ks}
    latencies: list[float] = []
    query_results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for item in items:
        relevant = [
            make_chunk_key(label["source_name"], label["chunk_index"])
            for label in item["relevant_chunks"]
        ]
        started = time.perf_counter()
        try:
            documents = service.search(
                item["question"],
                final_k=max_k,
                use_reranker=use_reranker,
                strict_reranker=use_reranker,
            )
            latency_ms = (time.perf_counter() - started) * 1000
            latencies.append(latency_ms)
            retrieved = [
                make_chunk_key(doc.metadata["source_name"], doc.metadata["chunk_index"])
                for doc in documents
            ]
            per_k: dict[str, dict[str, float]] = {}
            for k in ks:
                metrics = calculate_metrics(retrieved, relevant, k)
                metric_buckets[k].append(metrics)
                per_k[str(k)] = metrics.to_dict()

            query_results.append(
                {
                    "id": item["id"],
                    "category": item.get("category", ""),
                    "question": item["question"],
                    "relevant_chunks": item["relevant_chunks"],
                    "retrieved_chunks": [
                        {
                            "rank": rank,
                            "source_name": key[0],
                            "chunk_index": key[1],
                            "preview": documents[rank - 1].page_content[:160],
                        }
                        for rank, key in enumerate(retrieved, start=1)
                    ],
                    "latency_ms": latency_ms,
                    "metrics": per_k,
                }
            )
        except Exception as error:  # keep the full run useful when one request fails
            errors.append({"id": item["id"], "question": item["question"], "error": str(error)})

    summary = {str(k): mean_metrics(metric_buckets[k]) for k in ks}
    top3 = summary.get("3", {})
    passed = not errors and all(top3.get(name, 0.0) >= threshold for name, threshold in PASS_THRESHOLDS.items())
    return {
        "dataset_size": len(items),
        "evaluated_count": len(query_results),
        "ks": ks,
        "retrieval": " + ".join(
            [
                f"{rag_conf.get('embedding_provider', 'dashscope')} {rag_conf['embedding_model_name']}",
                "Chroma similarity search",
                (
                    f"{rag_conf['reranker_model_name']} rerank"
                    if use_reranker
                    else "no reranker"
                ),
            ]
        ),
        "reranker_enabled": use_reranker,
        "summary": summary,
        "latency_ms": {
            "average": statistics.fmean(latencies) if latencies else 0.0,
            "p95": percentile(latencies, 0.95),
            "measured_at_k": max_k,
        },
        "pass_thresholds_at_3": PASS_THRESHOLDS,
        "passed": passed,
        "errors": errors,
        "queries": query_results,
    }


def render_report(result: dict[str, Any], index_health: dict[str, Any]) -> str:
    lines = [
        "# 项目2检索评测报告",
        "",
        f"- 生成时间：{result['generated_at']}",
        f"- 数据集：{result['dataset_size']} 个问题，成功评测 {result['evaluated_count']} 个",
        f"- 检索链路：{result['retrieval']}",
        f"- 结论：{'通过' if result['passed'] else '未通过'}预设 Top3 基线",
        "",
        "## 索引健康",
        "",
        f"- Chunk 总数：{index_health['total_chunks']}",
        f"- 来源数：{len(index_health['source_counts'])}",
        f"- 缺少评测元数据：{index_health['missing_metadata']}",
        "",
        "## 汇总指标",
        "",
        "| K | Precision@K | Recall@K | HitRate@K | MRR@K | NDCG@K |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for k in result["ks"]:
        metrics = result["summary"][str(k)]
        lines.append(
            f"| {k} | {metrics['precision_at_k']:.3f} | {metrics['recall_at_k']:.3f} | "
            f"{metrics['hit_rate_at_k']:.3f} | {metrics['mrr_at_k']:.3f} | {metrics['ndcg_at_k']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## 延迟",
            "",
            f"- Top{result['latency_ms']['measured_at_k']} 平均端到端检索延迟：{result['latency_ms']['average']:.1f} ms",
            f"- Top{result['latency_ms']['measured_at_k']} P95端到端检索延迟：{result['latency_ms']['p95']:.1f} ms",
            "",
            "## Top3通过标准",
            "",
        ]
    )
    for name, threshold in result["pass_thresholds_at_3"].items():
        actual = result["summary"]["3"].get(name, 0.0)
        lines.append(f"- {name}: {actual:.3f} / {threshold:.2f}（{'通过' if actual >= threshold else '未通过'}）")

    misses = [query for query in result["queries"] if query["metrics"]["3"]["hit_rate_at_k"] == 0.0]
    lines.extend(["", "## Top3未命中问题", ""])
    if not misses:
        lines.append("无。")
    else:
        for query in misses:
            retrieved = ", ".join(
                f"{item['source_name']}#{item['chunk_index']}" for item in query["retrieved_chunks"][:3]
            )
            lines.append(f"- `{query['id']}` {query['question']}；实际召回：{retrieved}")

    if result["errors"]:
        lines.extend(["", "## 执行错误", ""])
        lines.extend(f"- `{item['id']}`：{item['error']}" for item in result["errors"])
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    ks = sorted(set(args.ks))
    if not ks or any(k <= 0 for k in ks) or 3 not in ks:
        raise ValueError("--ks must contain positive values and include the production baseline K=3")

    dataset_path = args.dataset.resolve()
    output_root = args.output_dir.resolve()
    items = load_dataset(dataset_path)
    service = VectorStoreService()
    available, source_counts = available_chunk_keys(service.vector_store)
    raw_records = service.vector_store.get(include=["metadatas"])
    total_chunks = len(raw_records.get("ids") or [])
    index_health = {
        "total_chunks": total_chunks,
        "source_counts": source_counts,
        "missing_metadata": total_chunks - len(available),
    }
    if index_health["missing_metadata"]:
        raise ValueError("current Chroma index contains chunks without source_name/chunk_index")
    validate_labels(items, available)

    use_reranker = (
        bool(rag_conf.get("reranker_enabled", False))
        if args.reranker == "config"
        else args.reranker == "on"
    )

    result = evaluate(
        service,
        items,
        ks,
        warmup=not args.skip_warmup,
        use_reranker=use_reranker,
    )
    result["generated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    result["dataset"] = str(dataset_path.relative_to(PROJECT_ROOT))
    result["index_health"] = index_health

    run_dir = output_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "report.md").write_text(render_report(result, index_health), encoding="utf-8")

    print(f"report={run_dir / 'report.md'}")
    print(json.dumps({"passed": result["passed"], "summary": result["summary"]}, ensure_ascii=False, indent=2))
    return 0 if not result["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
