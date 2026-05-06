"""
make_slide_figures_v6.py — high-DPI PNG output, with rigorous file accounting.

Changes from v5:
- safe_reproject() now returns a (gdf, status) tuple so we can track WHY each
  file was skipped (no CRS + no bounds, read failure, reproject failure, etc.).
- fig_overview() prints a precise breakdown to stdout so the numbers reported
  on the slide title match exactly what the script processed. No more guessing
  whether "115 plotted" really means 115.

File names are identical to v5, so slides_v3.tex compiles unchanged.

Run from ~/Desktop/shoreline_recon/ after audit_shorelines_v2.py has produced
audit_results/shapefile_audit.csv.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import pandas as pd
import geopandas as gpd

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Style — DPI bumped to 400 for crisp projector display
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 17,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "savefig.bbox": "tight",
    "savefig.dpi": 400,        # <-- the key change
})

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent.parent
DATA_ROOT = HERE / "data"
AUDIT_DIR = HERE / "audit_results"
AUDIT_CSV = AUDIT_DIR / "shapefile_audit.csv"
ANALYSIS_DIR = HERE / "analysis_results"
OUT_DIR = HERE / "slide_figs"
OUT_DIR.mkdir(exist_ok=True)

TARGET_CRS = "EPSG:32119"


def load_audit() -> pd.DataFrame:
    return pd.read_csv(AUDIT_CSV)


def safe_reproject(rel_path: str):
    """Read and reproject a shapefile to TARGET_CRS.

    Returns (gdf, status). status is one of:
      'ok'           — reprojected from a CRS the file declared
      'inferred'     — file had no CRS metadata; assumed EPSG:4269 from bounds
      'read_failed'  — geopandas could not read the file
      'empty'        — file read OK but had no features
      'no_crs_no_bounds' — no CRS metadata AND geometry was empty
      'crs_unguessable'  — no CRS metadata AND bounds outside lat/lon range
      'reproject_failed' — reprojection raised an exception
    """
    shp = DATA_ROOT / rel_path
    try:
        gdf = gpd.read_file(shp)
    except Exception:
        return None, "read_failed"
    if gdf.empty:
        return None, "empty"

    inferred = False
    if gdf.crs is None:
        bx = gdf.total_bounds
        # bounds may itself be NaN if geometry rows are all empty
        if any(pd.isna(v) for v in bx):
            return None, "no_crs_no_bounds"
        if -180 <= bx[0] <= 180 and -90 <= bx[1] <= 90:
            gdf = gdf.set_crs("EPSG:4269")
            inferred = True
        else:
            return None, "crs_unguessable"
    try:
        out = gdf.to_crs(TARGET_CRS)
    except Exception:
        return None, "reproject_failed"
    return out, "inferred" if inferred else "ok"


# ---------------------------------------------------------------------------
# Fig 1 — full overview (mirrors audit_shorelines_v2.py's overview plot)
# ---------------------------------------------------------------------------
def fig_overview(df: pd.DataFrame):
    print("Generating fig01_overview.png ...")
    cats = ("shoreline", "shoreline_composite", "shoreline_appended")
    sl = df[df["category"].isin(cats)].copy()
    valid = sl.dropna(subset=["primary_year"])

    fig, ax = plt.subplots(figsize=(13, 9))

    ymin, ymax = valid["primary_year"].min(), valid["primary_year"].max()
    norm = mcolors.Normalize(vmin=ymin, vmax=ymax)
    cmap = cm.viridis

    # Track each file's outcome precisely
    status_counts = {"ok": 0, "inferred": 0, "read_failed": 0, "empty": 0,
                     "no_crs_no_bounds": 0, "crs_unguessable": 0,
                     "reproject_failed": 0, "plot_failed": 0}

    for _, row in sl.iterrows():
        gdf, status = safe_reproject(row["path"])
        if gdf is None:
            status_counts[status] += 1
            continue

        yr = row["primary_year"]
        color = cmap(norm(yr)) if pd.notna(yr) else (0.7, 0.7, 0.7, 1.0)

        try:
            gdf.plot(ax=ax, color=color, linewidth=0.8, alpha=0.6)
            status_counts[status] += 1
        except Exception:
            status_counts["plot_failed"] += 1

    plotted = status_counts["ok"] + status_counts["inferred"]
    skipped = sum(v for k, v in status_counts.items()
                  if k not in ("ok", "inferred"))

    # Print precise accounting so the user knows exactly what's on the figure
    print(f"\n  File accounting for fig01_overview.png:")
    print(f"    Total shoreline-like files:       {len(sl)}")
    print(f"    Reprojected from declared CRS:    {status_counts['ok']}")
    print(f"    CRS inferred from bounds (\u2192EPSG:4269): {status_counts['inferred']}")
    print(f"    Plotted total:                    {plotted}")
    print(f"    Skipped:                          {skipped}")
    for k, v in status_counts.items():
        if k not in ("ok", "inferred") and v > 0:
            print(f"      \u2022 {k}: {v}")
    print()

    # Overlay baselines (red dashed)
    for _, row in df[df["category"] == "baseline"].iterrows():
        gdf, _ = safe_reproject(row["path"])
        if gdf is not None and not gdf.empty:
            try:
                gdf.plot(ax=ax, color="red", linewidth=1.5,
                         linestyle="--", alpha=0.7)
            except Exception:
                pass

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Shoreline year", fontsize=13)

    # Title now reports the exact precise breakdown
    ax.set_title(
        f"All NC coast shorelines (Moore et al. 2013 dataset)\n"
        f"({plotted} plotted: {status_counts['ok']} from declared CRS, "
        f"{status_counts['inferred']} inferred; {skipped} skipped)"
    )
    ax.set_xlabel(f"Easting (m, {TARGET_CRS})")
    ax.set_ylabel(f"Northing (m, {TARGET_CRS})")
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig01_overview.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 2 — coverage by year
# ---------------------------------------------------------------------------
def fig_coverage_by_year(df: pd.DataFrame):
    print("Generating fig02_coverage_by_year.png ...")
    sl = df[df["category"].isin(["shoreline", "shoreline_composite", "shoreline_appended"])]
    sl = sl[sl["path"].str.contains("reprojected", case=False, na=False)]
    sl = sl.dropna(subset=["primary_year"]).copy()
    sl["primary_year"] = sl["primary_year"].astype(int)

    counts = sl.groupby("primary_year").size().reset_index(name="n")

    fig, ax = plt.subplots(figsize=(11, 4.8))
    colors = ["#3a7bd5" if y < 1974 else "#d9633b" for y in counts["primary_year"]]
    ax.bar(counts["primary_year"], counts["n"], color=colors,
           width=2.0, edgecolor="white")

    ax.axvline(1974, color="black", linestyle=":", lw=1.2, alpha=0.6)
    ax.text(1974.5, ax.get_ylim()[1] * 0.92, "1974\nbreakpoint",
            fontsize=11, va="top")

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="#3a7bd5", label="Historical (pre-1974)"),
        Patch(color="#d9633b", label="Recent (1974\u20132004)"),
    ], loc="upper left", frameon=False)

    ax.set_xlabel("Year")
    ax.set_ylabel("Shoreline files")
    ax.set_title("Shoreline coverage by year (curated subset)")
    ax.set_xlim(1845, 2010)
    ax.set_xticks(range(1850, 2011, 20))

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig02_coverage_by_year.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 3 — CRS breakdown
# ---------------------------------------------------------------------------
def fig_crs_breakdown(df: pd.DataFrame):
    print("Generating fig03_crs_breakdown.png ...")
    sl = df[df["category"].isin(["shoreline", "shoreline_composite", "shoreline_appended"])].copy()

    def short_crs(c):
        if pd.isna(c) or c == "None" or c == "UNDEFINED":
            return "No CRS metadata"
        c = str(c)
        if "EPSG:32119" in c:
            return "EPSG:32119\n(NC State Plane, m)"
        if "EPSG:4269" in c:
            return "EPSG:4269\n(NAD83 lat/lon)"
        if "ftUS" in c:
            return "NC State Plane\n(US Feet)"
        return c[:30]

    sl["crs_short"] = sl["crs"].apply(short_crs)
    counts = sl["crs_short"].value_counts()

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = []
    for label in counts.index:
        if "EPSG:32119" in label:
            colors.append("#2e8b57")
        elif "No CRS" in label:
            colors.append("#c0392b")
        else:
            colors.append("#7f8c8d")

    bars = ax.barh(counts.index, counts.values, color=colors, edgecolor="white")
    for bar, count in zip(bars, counts.values):
        ax.text(count + 1, bar.get_y() + bar.get_height()/2, f"{count}",
                va="center", fontsize=12)

    ax.set_xlabel("Shoreline files")
    ax.set_title("Coordinate reference systems are heterogeneous")
    ax.invert_yaxis()
    ax.set_xlim(0, counts.values.max() * 1.18)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig03_crs_breakdown.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 4, 5 — Cape zooms
# ---------------------------------------------------------------------------
def fig_cape_zoom(df: pd.DataFrame, cape_name: str, dsas_dir_keyword: str,
                  bbox: tuple, output_name: str):
    print(f"Generating {output_name} ...")
    sl = df[df["category"].isin(["shoreline", "shoreline_composite", "shoreline_appended"])]
    sl = sl[sl["path"].str.contains("reprojected", case=False, na=False)]
    sl = sl[sl["path"].str.contains(dsas_dir_keyword, case=False, na=False)]
    sl = sl.dropna(subset=["primary_year"]).copy()

    fig, ax = plt.subplots(figsize=(9, 8))

    if sl.empty:
        ax.text(0.5, 0.5, f"No reprojected shorelines for {cape_name}",
                ha="center", va="center", transform=ax.transAxes)
        fig.savefig(OUT_DIR / output_name)
        plt.close(fig)
        return

    ymin, ymax = sl["primary_year"].min(), sl["primary_year"].max()
    norm = mcolors.Normalize(vmin=ymin, vmax=ymax)
    cmap = cm.viridis

    plotted = 0
    for _, row in sl.iterrows():
        gdf, _ = safe_reproject(row["path"])
        if gdf is None or gdf.empty:
            continue
        try:
            gdf = gdf.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        except Exception:
            pass
        if gdf.empty:
            continue
        gdf.plot(ax=ax, color=cmap(norm(row["primary_year"])),
                 linewidth=1.4, alpha=0.85)
        plotted += 1

    baseline_plotted = False
    for _, row in df[df["category"] == "baseline"].iterrows():
        if dsas_dir_keyword not in row["path"]:
            continue
        gdf, _ = safe_reproject(row["path"])
        if gdf is None or gdf.empty:
            continue
        try:
            gdf = gdf.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        except Exception:
            pass
        if not gdf.empty:
            gdf.plot(ax=ax, color="red", linewidth=1.5, linestyle="--", alpha=0.7)
            baseline_plotted = True

    ax.set_xlim(bbox[0], bbox[2])
    ax.set_ylim(bbox[1], bbox[3])

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Year", fontsize=13)

    ax.set_title(f"{cape_name} \u2014 {plotted} shorelines (curated subset)")
    ax.set_xlabel("Easting (m, EPSG:32119)")
    ax.set_ylabel("Northing (m, EPSG:32119)")

    if baseline_plotted:
        legend_handles = [Line2D([0], [0], color="red", lw=1.5, linestyle="--",
                                  label="DSAS baseline")]
        ax.legend(handles=legend_handles, loc="lower right",
                  frameon=True, framealpha=0.9, fontsize=11)

    fig.tight_layout()
    fig.savefig(OUT_DIR / output_name)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 6 — LRR replication, no y-axis clipping
# ---------------------------------------------------------------------------
def fig_lrr_replication():
    print("Generating fig06_lrr_replication.png ...")

    candidates = [
        ("Cape Hatteras: 1974\u20132004 (recent)",
         ANALYSIS_DIR / "Export74_04stand.xlsx", "#d9633b"),
        ("Cape Lookout: 1849\u20131973 (historical)",
         ANALYSIS_DIR / "Export_1849_1973.xlsx", "#3a7bd5"),
        ("Cape Lookout: 1973\u20132004 (recent)",
         ANALYSIS_DIR / "Export_t1973_2004.xlsx", "#d9633b"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(11, 8.5), sharex=False)

    for ax, (label, path, color) in zip(axes, candidates):
        if not path.exists():
            ax.text(0.5, 0.5, f"File not found:\n{path.name}",
                    transform=ax.transAxes, ha="center", va="center")
            ax.set_title(label)
            continue
        try:
            df = pd.read_excel(path)
        except Exception as e:
            ax.text(0.5, 0.5, f"Error reading: {e}",
                    transform=ax.transAxes, ha="center", va="center")
            continue

        lrr_col = next((c for c in df.columns if str(c).upper() == "LRR"), None)
        x_col = next((c for c in df.columns
                      if str(c).upper() in ("TRANSORDER", "OBJECTID")), None)

        if lrr_col is None:
            ax.text(0.5, 0.5, "No LRR column", transform=ax.transAxes,
                    ha="center", va="center")
            continue

        x = df[x_col] if x_col is not None else range(len(df))
        y = df[lrr_col]

        ax.plot(x, y, color=color, lw=1.0, alpha=0.85)
        ax.axhline(0, color="black", lw=0.6, alpha=0.5)
        ax.fill_between(x, 0, y, where=(y > 0), color="green", alpha=0.18,
                        label="accretion")
        ax.fill_between(x, 0, y, where=(y < 0), color="firebrick", alpha=0.18,
                        label="erosion")

        ax.set_title(label)
        ax.set_ylabel("LRR (m/yr)")
        ax.legend(loc="upper right", frameon=False, fontsize=10)

    axes[-1].set_xlabel("Transect order (alongshore)")
    fig.suptitle("Moore et al. 2013 LRR results, recovered from existing exports",
                 y=1.00)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig06_lrr_replication.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if not AUDIT_CSV.exists():
        print(f"ERROR: {AUDIT_CSV} not found.")
        print("Run audit_shorelines_v2.py first.")
        return

    df = load_audit()
    print(f"Loaded audit CSV: {len(df)} files\n")

    fig_overview(df)
    fig_coverage_by_year(df)
    fig_crs_breakdown(df)

    fig_cape_zoom(df, "Cape Hatteras", "DSAS_cpHat",
                  bbox=(870_000, 95_000, 945_000, 325_000),
                  output_name="fig04_cape_hatteras.png")
    fig_cape_zoom(df, "Cape Lookout", "DSAS_cpLO",
                  bbox=(640_000, -10_000, 890_000, 180_000),
                  output_name="fig05_cape_lookout.png")

    fig_lrr_replication()

    print(f"\nAll done. PNGs at 400 DPI in: {OUT_DIR}")
    for f in sorted(OUT_DIR.glob("*.png")):
        size_mb = f.stat().st_size / 1e6
        print(f"  {f.name:35s} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
