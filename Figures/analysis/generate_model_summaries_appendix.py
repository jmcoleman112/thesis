#!/usr/bin/env python
"""
Generate the appendix longtable from research/model_summaries.csv.

The exported TeX includes every CSV row in order. For pose rows, the accuracy
columns use the Validation2 metrics so the appendix matches the pose evaluation
convention used elsewhere in the thesis.

Run:
  python Figures/analysis/generate_model_summaries_appendix.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


CSV_DEFAULT = Path(__file__).resolve().parents[2] / "research" / "model_summaries.csv"
OUTPUT_DEFAULT = Path(__file__).resolve().parents[2] / "Thesis" / "tables" / "model_summaries_appendix.tex"

OBJECT_ACCURACY_COLUMNS = [
    "mAP50",
    "mAP50-95",
    "Precision",
    "Recall",
]
POSE_ACCURACY_COLUMNS = [
    "Validation2 mAP50",
    "Validation2 mAP50-95",
    "Validation2 Precision",
    "Validation2 Recall",
]
SYSTEM_COLUMNS = [
    "FPS (avg)",
    "Latency ms",
    "GPU Util %",
    "CPU Util %",
    "RAM (GB)",
    "Power (W)",
    "Temp (?C)",
]
REQUIRED_COLUMNS = ["Model", "Location", *OBJECT_ACCURACY_COLUMNS, *SYSTEM_COLUMNS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the appendix model summary longtable from CSV.")
    parser.add_argument("--csv", type=Path, default=CSV_DEFAULT, help="Path to model_summaries.csv")
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT, help="TeX output path")
    return parser.parse_args()


def is_pose_row(row: dict[str, str]) -> bool:
    location = row["Location"].replace("\\", "/").lower()
    return "/pose/" in location


def format_value(raw: str, decimals: int) -> str:
    text = (raw or "").strip()
    if not text:
        return "--"
    value = float(text)
    return f"{value:.{decimals}f}"


def build_row(row: dict[str, str]) -> str:
    accuracy_columns = POSE_ACCURACY_COLUMNS if is_pose_row(row) else OBJECT_ACCURACY_COLUMNS

    if is_pose_row(row):
        missing_pose_columns = [column for column in POSE_ACCURACY_COLUMNS if not (row.get(column) or "").strip()]
        if missing_pose_columns:
            missing_text = ", ".join(missing_pose_columns)
            raise ValueError(f"Missing pose Validation2 values for {row['Model']}: {missing_text}")

    parts = [rf"\path|{row['Model']}|"]
    parts.extend(format_value(row[column], 3) for column in accuracy_columns)
    parts.extend(format_value(row[column], 2) for column in SYSTEM_COLUMNS)
    return " & ".join(parts) + r" \\"


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
        if missing:
            missing_text = ", ".join(missing)
            raise ValueError(f"CSV is missing required columns: {missing_text}")
        rows = list(reader)
    if not rows:
        raise ValueError("CSV did not contain any rows.")
    return rows


def build_table(rows: list[dict[str, str]]) -> str:
    header = r"""\begingroup
\tiny
\setlength{\LTleft}{0pt}
\setlength{\LTright}{0pt}
\setlength{\LTcapwidth}{\linewidth}
\setlength{\tabcolsep}{2.5pt}
\renewcommand{\arraystretch}{0.95}
\begin{longtable}{cccccccccccc}
\caption{Selected benchmark fields from \texttt{model\_summaries.csv}. Pose rows use the Validation2 accuracy fields.}\label{tab:model_summaries_appendix}\\
\toprule
\makecell[l]{\textbf{Model}} & \makecell[r]{\textbf{mAP50}} & \makecell[r]{\textbf{mAP50-95}} & \makecell[r]{\textbf{Precision}} & \makecell[r]{\textbf{Recall}} & \makecell[r]{\textbf{FPS}\\\textbf{(avg)}} & \makecell[r]{\textbf{Latency}\\\textbf{ms}} & \makecell[r]{\textbf{GPU}\\\textbf{Util \%}} & \makecell[r]{\textbf{CPU}\\\textbf{Util \%}} & \makecell[r]{\textbf{RAM}\\\textbf{(GB)}} & \makecell[r]{\textbf{Power}\\\textbf{(W)}} & \makecell[r]{\textbf{Temp}\\\textbf{(C)}} \\
\midrule
\endfirsthead
\multicolumn{12}{l}{\small\itshape Table \thetable\ continued from previous page.}\\
\toprule
\makecell[l]{\textbf{Model}} & \makecell[r]{\textbf{mAP50}} & \makecell[r]{\textbf{mAP50-95}} & \makecell[r]{\textbf{Precision}} & \makecell[r]{\textbf{Recall}} & \makecell[r]{\textbf{FPS}\\\textbf{(avg)}} & \makecell[r]{\textbf{Latency}\\\textbf{ms}} & \makecell[r]{\textbf{GPU}\\\textbf{Util \%}} & \makecell[r]{\textbf{CPU}\\\textbf{Util \%}} & \makecell[r]{\textbf{RAM}\\\textbf{(GB)}} & \makecell[r]{\textbf{Power}\\\textbf{(W)}} & \makecell[r]{\textbf{Temp}\\\textbf{(C)}} \\
\midrule
\endhead
\midrule
\multicolumn{12}{r}{\small\itshape Continued on next page.}\\
\endfoot
\bottomrule
\endlastfoot
"""
    body = "\n".join(build_row(row) for row in rows)
    footer = r"""
\end{longtable}
\endgroup
"""
    return header + body + footer


def main() -> int:
    args = parse_args()
    rows = load_rows(args.csv)
    output_text = build_table(rows)
    args.output.write_text(output_text, encoding="utf-8", newline="\n")
    print(f"Wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
