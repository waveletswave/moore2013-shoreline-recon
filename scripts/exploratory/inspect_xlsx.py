"""
Inspect all the xlsx files from the Moore et al. 2013 dataset.

Run on your local Mac:
    cd ~/Desktop/shoreline_recon/analysis_results
    python3 inspect_xlsx.py

This script:
- Lists every sheet in every xlsx
- Shows the first 5 rows + column names of each sheet
- Highlights any column that looks like LRR / shoreline change rate
- Prints a summary table of file sizes and shapes

No modifications to your files. Read-only.
"""

import os
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    print("ERROR: pandas not installed. Run: pip3 install pandas openpyxl")
    raise SystemExit(1)

HERE = Path(__file__).resolve().parent.parent.parent
ANALYSIS_DIR = HERE / "analysis_results"
XLSX_FILES = sorted(ANALYSIS_DIR.glob("*.xlsx"))

if not XLSX_FILES:
    print(f"No .xlsx files found in {ANALYSIS_DIR}")
    print("Make sure you run this from ~/Desktop/shoreline_recon/analysis_results")
    raise SystemExit(1)

print("=" * 70)
print(f"Found {len(XLSX_FILES)} xlsx files in {ANALYSIS_DIR}")
print("=" * 70)
for f in XLSX_FILES:
    print(f"  {f.name}  ({f.stat().st_size // 1024} KB)")
print()


# Inspect each file
for f in XLSX_FILES:
    print()
    print("#" * 70)
    print(f"# FILE: {f.name}")
    print("#" * 70)
    
    try:
        xl = pd.ExcelFile(f)
        sheets = xl.sheet_names
        print(f"Sheets ({len(sheets)}): {sheets}")
        
        for sheet in sheets:
            print(f"\n--- Sheet: {sheet!r} ---")
            df = xl.parse(sheet)
            print(f"  Shape: {df.shape[0]} rows x {df.shape[1]} cols")
            print(f"  Columns: {list(df.columns)}")
            
            # Highlight LRR / change-rate columns
            rate_cols = [c for c in df.columns 
                         if any(k in str(c).upper() 
                                for k in ['LRR', 'RATE', 'CHANGE', 'EPR', 'NSM'])]
            if rate_cols:
                print(f"  ⭐ Likely rate columns: {rate_cols}")
            
            # Show first 5 rows (truncated for width)
            with pd.option_context('display.max_columns', 8,
                                   'display.width', 120,
                                   'display.max_colwidth', 20):
                print("  First 5 rows:")
                print(df.head().to_string(index=False).replace('\n', '\n    '))
    
    except Exception as e:
        print(f"  ERROR reading {f.name}: {e}")

print()
print("=" * 70)
print("Done. Read through the output to understand the data structure.")
print("=" * 70)
