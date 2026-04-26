"""CI eval gate: run kosik-eval-golden, check thresholds, emit markdown report.

Volá `run_evaluation()` (`src/kosik_workshop/evals/runner.py`) — žádná nová eval
logika. Po doběhnutí spočítá průměrné skóre per evaluator, srovná s thresholdem
a zapíše markdown report do souboru pro PR komentář (a do `$GITHUB_STEP_SUMMARY`
pokud běží v Actions).

Exit code:
    0 — všechny evaluators ≥ threshold
    1 — alespoň jeden propadl

Usage (lokálně):
    uv run python scripts/ci_eval_gate.py --threshold 0.8 --report-file report.md
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from kosik_workshop.agent import build_agent
from kosik_workshop.config import load_env
from kosik_workshop.evals.dataset import DATASET_NAME
from kosik_workshop.evals.runner import run_evaluation


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--threshold", type=float, default=0.8, help="Min score per evaluator (default 0.8)."
    )
    p.add_argument("--report-file", type=Path, required=True, help="Markdown report destination.")
    p.add_argument("--model", default="gpt-4o-mini", help="Model passed to build_agent().")
    return p.parse_args()


def _pr_number_from_ref(ref: str) -> str | None:
    # GITHUB_REF na pull_request je "refs/pull/<n>/merge"
    parts = ref.split("/") if ref else []
    if len(parts) >= 3 and parts[1] == "pull":
        return parts[2]
    return None


def _ci_context() -> dict[str, str]:
    sha = os.getenv("GITHUB_SHA", "local")
    ref = os.getenv("GITHUB_REF", "")
    branch = os.getenv("GITHUB_REF_NAME", "local")
    pr = _pr_number_from_ref(ref) or os.getenv("PR_NUMBER", "0")
    return {"sha": sha, "branch": branch, "pr": pr}


def _experiment_url(experiment_name: str) -> str:
    base = os.getenv("LANGSMITH_ENDPOINT", "https://eu.smith.langchain.com").rstrip("/")
    base = base.replace("api.", "").replace("/api", "")
    return f"{base}/o/-/datasets?search={DATASET_NAME}#experiments={experiment_name}"


def _build_report(
    *,
    scores: dict[str, float],
    threshold: float,
    ctx: dict[str, str],
    model: str,
    n_examples: int,
    experiment_name: str,
) -> tuple[str, bool]:
    lines: list[str] = []
    lines.append(f"## 🤖 Eval Gate — `{DATASET_NAME}`")
    lines.append("")
    lines.append(
        f"Commit `{ctx['sha'][:7]}` · {n_examples} příkladů · model `{model}` · "
        f"experiment `{experiment_name}`"
    )
    lines.append("")
    lines.append("| Evaluator | Score | Threshold | Status |")
    lines.append("|---|---|---|---|")

    failed: list[str] = []
    for name in sorted(scores):
        score = scores[name]
        status = "✅" if score >= threshold else "❌"
        if score < threshold:
            failed.append(f"`{name}` = {score:.3f}")
        lines.append(f"| `{name}` | {score:.3f} | ≥ {threshold:.2f} | {status} |")

    lines.append("")
    lines.append(f"[→ View experiment in LangSmith]({_experiment_url(experiment_name)})")
    lines.append("")
    if failed:
        lines.append(f"❌ **Failed:** {', '.join(failed)} (threshold {threshold:.2f}).")
    else:
        lines.append(f"✅ **Passed:** všechny evaluators ≥ {threshold:.2f}.")
    lines.append("")
    return "\n".join(lines), bool(failed)


def main() -> int:
    args = parse_args()
    load_env()

    ctx = _ci_context()
    sha_short = ctx["sha"][:7] if ctx["sha"] != "local" else "local"
    experiment_prefix = f"ci-pr{ctx['pr']}-{sha_short}"

    print(f"Running eval gate (threshold={args.threshold}, prefix={experiment_prefix})")

    agent = build_agent(model=args.model)
    result: Any = run_evaluation(
        agent,
        experiment_prefix=experiment_prefix,
        metadata={
            "ci": True,
            "sha": ctx["sha"],
            "pr": ctx["pr"],
            "branch": ctx["branch"],
            "model": args.model,
        },
        description=f"CI eval gate · PR #{ctx['pr']} · {sha_short}",
    )

    df = result.to_pandas()
    score_cols = [c for c in df.columns if c.startswith("feedback.")]
    if not score_cols:
        print("error: no feedback.* columns in eval result", file=sys.stderr)
        return 1

    scores = {col.removeprefix("feedback."): float(df[col].mean()) for col in score_cols}
    experiment_name = getattr(result, "experiment_name", experiment_prefix)

    report, failed = _build_report(
        scores=scores,
        threshold=args.threshold,
        ctx=ctx,
        model=args.model,
        n_examples=len(df),
        experiment_name=experiment_name,
    )

    args.report_file.write_text(report, encoding="utf-8")
    print(f"Report → {args.report_file}")

    summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as fh:
            fh.write(report + "\n")

    print()
    print(report)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
