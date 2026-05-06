"""
make_fig09_data_availability_v2.py — redesigned data availability figure.

Replaces the scatter+jitter design with a clean heatmap grid:
  - Each column = one year (only years that exist appear)
  - Each row = (region, tier) combination, grouped visually by region
  - Cell = number of files for that (year, region, tier)
  - Cell color = tier color (green / yellow / orange / red)
  - Cell text = file count (only when >= 1)

This is faster to read in a lab meeting because every cell answers a
specific question: "do we have year X data in region Y at processing
difficulty Z, and how many files?"

Run from ~/Desktop/shoreline_recon/.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "axes.grid": False,
    "savefig.bbox": "tight",
    "savefig.dpi": 400,
})

HERE = Path(__file__).resolve().parent.parent
AUDIT_CSV = HERE / "audit_results" / "shapefile_audit.csv"
OUT_DIR = HERE / "slide_figs"
OUT_DIR.mkdir(exist_ok=True)


def classify_tier(row):
    path = str(row["path"]).lower()
    crs = str(row["crs"])
    if "reprojected" in path:
        return "1_ready"
    if pd.isna(row["crs"]) or crs in ("None", "UNDEFINED"):
        return "4_significant"
    if "ftUS" in crs:
        return "3_moderate"
    if "EPSG:4269" in crs or "EPSG:32119" in crs:
        return "2_minor"
    return "5_other"


def classify_region(p):
    p = str(p).lower()
    if "cphat" in p or "cape_hat" in p or "hatteras" in p:
        return "Cape Hatteras"
    if "cplo" in p or "cape_lo" in p or "lookout" in p:
        return "Cape Lookout"
    return "Other / mixed"


# Tier metadata: (display label, fill color, text color)
TIER_META = {
    "1_ready":       ("Ready",         "#2e8b57", "white"),
    "2_minor":       ("Minor prep",    "#e8a838", "black"),
    "3_moderate":    ("Moderate prep", "#d9633b", "white"),
    "4_significant": ("Significant",   "#c0392b", "white"),
}
TIER_ORDER = ["1_ready", "2_minor", "3_moderate", "4_significant"]
REGION_ORDER = ["Cape Hatteras", "Cape Lookout", "Other / mixed"]


def main():
    df = pd.read_csv(AUDIT_CSV)
    sl = df[df["category"].isin(
        ["shoreline", "shoreline_composite", "shoreline_appended"])].copy()
    sl = sl.dropna(subset=["primary_year"]).copy()
    sl["primary_year"] = sl["primary_year"].astype(int)
    sl["tier"] = sl.apply(classify_tier, axis=1)
    sl["region"] = sl["path"].apply(classify_region)

    years = sorted(sl["primary_year"].unique())

    # Build row labels: for each region, list its tiers in order
    rows = []
    for region in REGION_ORDER:
        for tier in TIER_ORDER:
            rows.append((region, tier))

    # Count files per (year, region, tier)
    counts = (sl.groupby(["primary_year", "region", "tier"])
                .size().reset_index(name="n"))

    # Build the grid: rows × columns
    grid = np.zeros((len(rows), len(years)), dtype=int)
    for _, r in counts.iterrows():
        try:
            i = rows.index((r["region"], r["tier"]))
            j = years.index(r["primary_year"])
            grid[i, j] = r["n"]
        except ValueError:
            continue

    # Plot
    fig, ax = plt.subplots(figsize=(15, 6.5))

    # Draw cells one by one (so each can have its tier color)
    for i, (region, tier) in enumerate(rows):
        color = TIER_META[tier][1]
        text_color = TIER_META[tier][2]
        for j, yr in enumerate(years):
            n = grid[i, j]
            if n > 0:
                ax.add_patch(plt.Rectangle(
                    (j - 0.45, i - 0.45), 0.9, 0.9,
                    facecolor=color, edgecolor="white", linewidth=1.5))
                ax.text(j, i, str(n), ha="center", va="center",
                        color=text_color, fontsize=11, fontweight="bold")
            else:
                # Empty cell: very light gray background
                ax.add_patch(plt.Rectangle(
                    (j - 0.45, i - 0.45), 0.9, 0.9,
                    facecolor="#f5f5f5", edgecolor="white", linewidth=1.5))

    # Y-axis: combine region prefix with tier label, e.g. "Hatteras: Ready"
    region_short = {
        "Cape Hatteras": "Hatteras",
        "Cape Lookout": "Lookout",
        "Other / mixed": "Other",
    }
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(
        [f"{region_short[region]}: {TIER_META[tier][0]}"
         for region, tier in rows]
    )
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.invert_yaxis()

    # Horizontal separators between regions
    for i in [4, 8]:
        ax.axhline(i - 0.5, color="black", lw=1.0, alpha=0.3)

    # X-axis: years
    ax.set_xticks(range(len(years)))
    ax.set_xticklabels(years, rotation=45, ha="right")
    ax.set_xlim(-0.6, len(years) - 0.4)

    # 1974 breakpoint
    if 1974 in years:
        bp = years.index(1974)
        ax.axvline(bp - 0.5, color="black", linestyle="--", lw=1.2,
                   alpha=0.6, zorder=10)
        ax.text(bp - 0.5, -1.1, "1974 breakpoint",
                fontsize=10, ha="center", color="#444444")

    ax.set_title(
        "Shoreline file availability by year, region, and processing difficulty\n"
        f"(every cell = number of files; total = {len(sl)})",
        pad=18,
    )

    # Remove default spines for cleaner look
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(left=False, bottom=False)

    # Legend at the bottom
    legend_handles = [
        mpatches.Patch(color=TIER_META[t][1],
                       label=f"{TIER_META[t][0]}  (n={(sl['tier']==t).sum()})")
        for t in TIER_ORDER
    ]
    ax.legend(handles=legend_handles,
              loc="lower center", bbox_to_anchor=(0.5, -0.28),
              ncol=4, frameon=False, fontsize=11)

    fig.tight_layout()
    out = OUT_DIR / "fig09_data_availability.png"
    fig.savefig(out)
    print(f"Saved: {out}")

    # Print the underlying numbers for reference
    print(f"\nGrid summary: {len(years)} year columns × {len(rows)} (region, tier) rows")
    print(f"Years: {years}")
    print()
    print("Files per (region, tier):")
    print(pd.crosstab(sl["region"], sl["tier"], margins=True))


if __name__ == "__main__":
    main()
