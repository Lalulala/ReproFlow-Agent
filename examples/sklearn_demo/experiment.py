"""Small, deterministic CPU experiment used by ReproFlow's end-to-end demo."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def build_model(name: str, seed: int):
    if name == "logistic_regression":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed))
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=120, max_depth=6, random_state=seed)
    if name == "svm":
        return make_pipeline(StandardScaler(), SVC(C=2.0, probability=True, random_state=seed))
    raise ValueError(f"Unsupported model: {name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--simulate-failure", action="store_true")
    parser.add_argument("--simulate-timeout", type=float, default=0.0)
    args = parser.parse_args()

    if args.simulate_failure:
        raise RuntimeError("Intentional demo failure")
    if args.simulate_timeout:
        time.sleep(args.simulate_timeout)

    data = load_breast_cancer()
    x_train, x_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=0.25,
        random_state=args.seed,
        stratify=data.target,
    )
    model = build_model(args.model, args.seed)
    started = time.perf_counter()
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    probability = model.predict_proba(x_test)[:, 1]
    duration = time.perf_counter() - started

    payload = {
        "model": args.model,
        "seed": args.seed,
        "accuracy": accuracy_score(y_test, prediction),
        "f1": f1_score(y_test, prediction),
        "roc_auc": roc_auc_score(y_test, probability),
        "duration_seconds": duration,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "pid": os.getpid(),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print("METRICS_JSON=" + json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

