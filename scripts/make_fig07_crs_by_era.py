"""
make_fig07_crs_by_era.py — generate just the CRS-by-era backup figure.

Run from ~/Desktop/shoreline_recon/. Adds fig07_crs_by_era.png to slide_figs/.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "savefig.bbox": "tight",
    "savefig.dpi": 400,
})

HERE = Path(__file__).resolve().parent.parent
AUDIT_CSV = HERE / "audit_results" / "shapefile_audit.csv"
OUT_DIR = HERE / "slide_figs"
OUT_DIR.mkdir(exist_ok=True)


def short_crs(c):
    if pd.isna(c) or str(c) in ("None", "UNDEFINED"):
        return "No CRS metadata"
    c = str(c)
    if "EPSG:32119" in c:
        return "EPSG:32119 (m)"
    if "EPSG:4269" in c:
        return "EPSG:4269 (lat/lon)"
    if "ftUS" in c:
        return "NC State Plane (ft)"
    return c[:25]


def era(y):
    if y < 1900: return "1849\u20131899"
    if y < 1950: return "1900\u20131949"
    if y < 1980: return "1950\u20131979"
    return "1980\u20132004"


def main():
    df = pd.read_csv(AUDIT_CSV)
    sl = df[df["category"].isin(["shoreline", "shoreline_composite", "shoreline_appended"])]
    # Use ORIGINAL files only — Laura's reprojected copies would mask the
    # historical pattern by mapping every era to EPSG:32119.
    sl = sl[~sl["path"].str.contains("reprojected", case=False, na=False)]
    sl = sl.dropna(subset=["primary_year"]).copy()
    sl["primary_year"] = sl["primary_year"].astype(int)
    sl["crs_short"] = sl["crs"].apply(short_crs)
    sl["era"] = sl["primary_year"].apply(era)

    # Order columns (CRSes) so the legend reads chronologically
    crs_order = [
        "EPSG:4269 (lat/lon)",
        "NC State Plane (ft)",
        "EPSG:32119 (m)",
        "No CRS metadata",
    ]
    era_order = ["1849\u20131899", "1900\u20131949", "1950\u20131979", "1980\u20132004"]

    ct = pd.crosstab(sl["era"], sl["crs_short"]).reindex(
        index=era_order, columns=crs_order, fill_value=0)

    # Stacked horizontal bar chart
    fig, ax = plt.subplots(figsize=(11, 5.2))
    color_map = {
        "EPSG:4269 (lat/lon)":   "#7f8c8d",   # gray (geographic, older)
        "NC State Plane (ft)":   "#d9633b",   # orange (mid-era)
        "EPSG:32119 (m)":        "#2e8b57",   # green (modern)
        "No CRS metadata":       "#c0392b",   # red (problem)
    }
    bottom = pd.Series([0]*len(era_order), index=era_order, dtype=float)
    for crs in crs_order:
        vals = ct[crs]
        ax.barh(era_order, vals, left=bottom, color=color_map[crs],
                edgecolor="white", label=crs)
        # Inline counts when bar segment >= 2
        for i, (era_label, v) in enumerate(vals.items()):
            if v >= 2:
                ax.text(bottom[era_label] + v/2, i, str(v),
                        ha="center", va="center", color="white",
                        fontsize=12, fontweight="bold")
        bottom = bottom + vals

    ax.set_xlabel("Number of original shoreline files")
    ax.set_title("CRS conventions changed with the data-collection era\n"
                 "(original files only; Laura's reprojected copies excluded)")
    ax.invert_yaxis()
    ax.legend(loc="upper right", frameon=True, framealpha=0.95)

    fig.tight_layout()
    out = OUT_DIR / "fig07_crs_by_era.png"
    fig.savefig(out)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
