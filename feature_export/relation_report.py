"""Reports render stored diagnostic scores; no metric recomputation here."""

from html import escape


def percent(value):
    return "undefined" if value is None else f"{100 * value:.6f}%"


def report_markdown(summary):
    lines = [f"# {summary['method']} class relation diagnostic: {summary['dataset']}", "",
             f"Eligible triplets: {summary['eligible_triplets']} / {summary['total_triplets']} "
             f"({percent(summary['coverage'])}).", "",
             "| Seed | Reversed | Reversal rate | Tied | Tie rate |",
             "|---|---:|---:|---:|---:|"]
    for row in summary["per_seed"]:
        lines.append(f"| {row['seed']} | {row['reversed_triplets']} | {percent(row['reversal_rate'])} | "
                     f"{row['tied_triplets']} | {percent(row['tie_rate'])} |")
    comparison = summary.get("comparison")
    if comparison:
        lines += ["", f"ZeroDiff: {comparison['zerodiff_reversed_triplets']} reversals, "
                  f"{percent(comparison['zerodiff_reversal_rate'])}; tie rate "
                  f"{percent(comparison['zerodiff_tie_rate'])}.",
                  f"Rediff minus ZeroDiff: {comparison['delta_percentage_points']:+.6f} percentage points."]
    lines += ["", f"Conclusion: `{summary['conclusion']}`.",
              "", "Features are averaged in float64 before cosine distances are calculated.",
              "Eligibility depends only on semantics and real visual centers from the verified reference.",
              "Ties are separate from reversals. Seeds control generation from fixed weights.",
              "A single generation seed has no cross-seed standard deviation."]
    return "\n".join(lines) + "\n"


def figure_svg(summary, rows, outcomes, class_names):
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="650" viewBox="0 0 1200 650">',
             '<rect width="1200" height="650" fill="white"/>',
             '<g font-family="Arial, sans-serif" fill="#17212b">']

    def text(x, y, value, size=16):
        parts.append(f'<text x="{x}" y="{y}" font-size="{size}">{escape(str(value))}</text>')

    text(30, 36, f"{summary['method']}: {summary['dataset']} unseen class relations", 25)
    text(30, 66, f"Eligible: {summary['eligible_triplets']} / {summary['total_triplets']}; "
         f"semantic and real relative gaps strictly greater than {summary['min_gap']}")
    text(30, 110, "Observed reversal example", 20)
    example = summary["example"]
    if example is None:
        text(30, 160, "No eligible reversed triplet was observed.")
    else:
        row = rows[example["triplet_index"]]
        outcome = outcomes[example["seed_index"]][example["triplet_index"]]
        for index, role in enumerate(("anchor", "near", "far")):
            # The full labels remain in triplets.csv; keep the plot within its panel.
            label = str(class_names[row[role]])
            text(30, 145 + index * 25, f"{role}: {label[:54]}", 14)
        spaces = [("Semantics", row["semantic_near"], row["semantic_far"]),
                  ("Real visual centers", row["real_near"], row["real_far"]),
                  ("Generated visual centers", outcome["generated_near"], outcome["generated_far"])]
        for index, (label, near, far) in enumerate(spaces):
            y = 235 + index * 95
            text(30, y, label)
            for offset, value, color, name in ((12, near, "#247ba0", "near"), (38, far, "#e59a34", "far")):
                width = 240 * value / max(near, far, 1e-12)
                text(30, y + offset + 15, name, 13)
                parts.append(f'<rect x="75" y="{y+offset}" width="{width:.3f}" height="18" fill="{color}"/>')
                text(330, y + offset + 15, f"{value:.7f}", 13)
        text(30, 550, f"Triplet {example['triplet_index']}; seed {summary['seeds'][example['seed_index']]}", 14)
        text(30, 575, "Each space uses its own distance scale.", 14)
    text(630, 110, "Measured reversal and tie rates", 20)
    comparison = summary.get("comparison")
    records = [("Rediff", r["reversal_rate"], r["tie_rate"]) for r in summary["per_seed"]]
    if comparison:
        records.insert(0, ("ZeroDiff", comparison["zerodiff_reversal_rate"], comparison["zerodiff_tie_rate"]))
    max_rate = max([v for _, reversal, tie in records for v in (reversal, tie) if v is not None] + [0.01])
    for index, (label, reversal, tie) in enumerate(records):
        y = 165 + index * (320 / max(len(records), 2))
        text(630, y, label, 18)
        for offset, value, color, name in ((18, reversal, "#247ba0", "reversal"), (50, tie, "#7f8c8d", "tie")):
            width = 300 * (value or 0) / max_rate
            parts.append(f'<rect x="710" y="{y+offset}" width="{width:.3f}" height="20" fill="{color}"/>')
            text(630, y + offset + 15, name, 13)
            text(1020, y + offset + 15, percent(value), 13)
    if comparison:
        text(630, 550, f"Change: {comparison['delta_percentage_points']:+.6f} percentage points", 17)
    text(30, 625, "Original 2048-D visual means; one-step inference; ties do not count as reversals.", 14)
    parts.append("</g></svg>")
    return "\n".join(parts)
