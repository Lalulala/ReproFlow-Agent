from pathlib import Path

from reproflow.evaluation import run_agent_evals

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    report = run_agent_evals(
        root / "evals" / "agent_cases.jsonl",
        output_path=root / "evals" / "latest_results.json",
    )
    print(
        f"Agent evals: {report['passed']}/{report['total']} passed; "
        f"threshold_met={report['threshold_met']}"
    )
    raise SystemExit(0 if report["threshold_met"] else 1)
