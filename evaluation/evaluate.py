"""Run small offline retrieval metrics against evaluation/dataset.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.main import SupportAssistant  # noqa: E402


def evaluate(dataset: Path = Path(__file__).with_name("dataset.json")) -> Dict[str, float]:
    rows: List[Dict[str, Any]] = json.loads(dataset.read_text(encoding="utf-8"))
    assistant = SupportAssistant(ROOT, llm_enabled=False)
    recalls, reciprocal = [], []
    for row in rows:
        result = assistant.answer(row["query"])
        ids = [source["id"] for source in result["sources"]]
        expected = set(row["relevant"])
        recalls.append(float(bool(expected.intersection(ids))))
        reciprocal.append(
            next((1.0 / (index + 1) for index, value in enumerate(ids) if value in expected), 0.0)
        )
    return {
        "queries": float(len(rows)),
        "Recall@5": round(sum(recalls) / max(1, len(recalls)), 3),
        "MRR": round(sum(reciprocal) / max(1, len(reciprocal)), 3),
    }


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2))
