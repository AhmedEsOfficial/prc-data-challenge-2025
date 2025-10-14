#!/usr/bin/env python3
"""
Parquet Data Explorer
======================
Analyzes parquet files in the project to understand their schema and contents.

Strategy:
- Analyze all .parquet files in the CWD
- Analyze ONE sample from flights_rank/ directory
- Analyze ONE sample from flights_train/ directory
"""

import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from typing import Dict
import sys
from io import StringIO


def print_separator(char="=", length=100):
    """Print a visual separator."""
    print(char * length)


def print_section_header(title: str):
    """Print a formatted section header."""
    print_separator("=")
    print(f"  {title}")
    print_separator("=")


def analyze_parquet_file(file_path: Path, show_sample_rows: int = 5) -> Dict:
    """
    Comprehensively analyze a single parquet file.

    Args:
        file_path: Path to the parquet file
        show_sample_rows: Number of sample rows to display

    Returns:
        Dictionary containing analysis results
    """
    print_section_header(f"FILE: {file_path}")

    try:
        # Read parquet file
        df = pd.read_parquet(file_path)

        # Get parquet metadata
        parquet_file = pq.ParquetFile(file_path)

        # Basic information
        print(f"\n📊 BASIC INFORMATION")
        print(f"  File Size: {file_path.stat().st_size / (1024*1024):.2f} MB")
        print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        print(f"  Memory Usage: {df.memory_usage(deep=True).sum() / (1024*1024):.2f} MB")
        print(f"  Number of Row Groups: {parquet_file.num_row_groups}")

        # Schema information
        print(f"\n📋 SCHEMA")
        print(f"  Column Name                    | Data Type          | Non-Null Count      | Null Count")
        print(f"  " + "-" * 95)
        for col in df.columns:
            non_null = df[col].count()
            null_count = df[col].isna().sum()
            print(f"  {col:30} | {str(df[col].dtype):18} | {non_null:15,} | {null_count:,}")

        # Statistical summary for numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            print(f"\n📈 NUMERIC COLUMN STATISTICS")
            print(df[numeric_cols].describe().to_string())

        # Categorical/Object columns summary
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            print(f"\n🏷️  CATEGORICAL COLUMN SUMMARY")
            for col in categorical_cols:
                unique_count = df[col].nunique()
                print(f"  {col}: {unique_count:,} unique values")
                if unique_count <= 10:
                    print(f"    Values: {df[col].unique().tolist()}")

        # Sample data
        print(f"\n📄 SAMPLE DATA (First {show_sample_rows} rows)")
        print(df.head(show_sample_rows).to_string())

        # Data quality checks
        print(f"\n✅ DATA QUALITY")
        total_cells = df.shape[0] * df.shape[1]
        total_nulls = df.isna().sum().sum()
        null_percentage = (total_nulls / total_cells * 100) if total_cells > 0 else 0
        print(f"  Total Missing Values: {total_nulls:,} ({null_percentage:.2f}%)")
        print(f"  Duplicate Rows: {df.duplicated().sum():,}")

        # Column-wise missing value details (if any)
        cols_with_nulls = df.columns[df.isna().any()].tolist()
        if cols_with_nulls:
            print(f"  Columns with Missing Values:")
            for col in cols_with_nulls:
                null_count = df[col].isna().sum()
                null_pct = (null_count / len(df) * 100)
                print(f"    - {col}: {null_count:,} ({null_pct:.2f}%)")

        print("\n")

        return {
            "file_path": str(file_path),
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.to_dict(),
            "total_nulls": total_nulls,
            "success": True
        }

    except Exception as e:
        print(f"❌ ERROR analyzing {file_path}: {str(e)}\n")
        return {
            "file_path": str(file_path),
            "success": False,
            "error": str(e)
        }


def get_sample_file_from_directory(directory: Path) -> Path:
    """
    Get the first parquet file from a directory (deterministic).

    Args:
        directory: Path to the directory

    Returns:
        Path to the first parquet file (sorted alphabetically)
    """
    parquet_files = sorted(directory.glob("*.parquet"))
    if parquet_files:
        return parquet_files[0]
    return None


def main():
    """Main analysis workflow."""

    # Base directory
    base_dir = Path(".")

    # Create analysis directory if it doesn't exist
    analysis_dir = base_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)

    # Output file
    output_file = analysis_dir / "parquet-results.md"

    # Redirect stdout to file
    original_stdout = sys.stdout
    sys.stdout = open(output_file, 'w', encoding='utf-8')

    print_section_header("PARQUET DATA ANALYSIS REPORT")
    print(f"Base Directory: {base_dir}")
    print(f"Generated: {pd.Timestamp.now()}")
    print()

    analysis_results = []

    # 1. Analyze all parquet files in the CWD
    print_section_header("SECTION 1: ROOT DIRECTORY PARQUET FILES")

    root_parquet_files = sorted([
        f for f in base_dir.glob("*.parquet")
        if f.is_file()
    ])

    print(f"Found {len(root_parquet_files)} parquet files in root directory:\n")
    for f in root_parquet_files:
        print(f"  - {f.name}")
    print()

    for parquet_file in root_parquet_files:
        result = analyze_parquet_file(parquet_file)
        analysis_results.append(result)

    # 2. Analyze ONE sample from flights_rank/
    print_section_header("SECTION 2: FLIGHTS_RANK DIRECTORY (SAMPLE)")

    flights_rank_dir = base_dir / "flights_rank"
    if flights_rank_dir.exists():
        sample_file = get_sample_file_from_directory(flights_rank_dir)
        if sample_file:
            total_files = len(list(flights_rank_dir.glob("*.parquet")))
            print(f"Directory contains {total_files:,} parquet files")
            print(f"Analyzing SAMPLE file (first alphabetically): {sample_file.name}\n")
            result = analyze_parquet_file(sample_file)
            analysis_results.append(result)
        else:
            print("❌ No parquet files found in flights_rank/\n")
    else:
        print("❌ flights_rank/ directory not found\n")

    # 3. Analyze ONE sample from flights_train/
    print_section_header("SECTION 3: FLIGHTS_TRAIN DIRECTORY (SAMPLE)")

    flights_train_dir = base_dir / "flights_train"
    if flights_train_dir.exists():
        sample_file = get_sample_file_from_directory(flights_train_dir)
        if sample_file:
            total_files = len(list(flights_train_dir.glob("*.parquet")))
            print(f"Directory contains {total_files:,} parquet files")
            print(f"Analyzing SAMPLE file (first alphabetically): {sample_file.name}\n")
            result = analyze_parquet_file(sample_file)
            analysis_results.append(result)
        else:
            print("❌ No parquet files found in flights_train/\n")
    else:
        print("❌ flights_train/ directory not found\n")

    # Summary
    print_section_header("ANALYSIS SUMMARY")
    successful = sum(1 for r in analysis_results if r.get("success", False))
    failed = len(analysis_results) - successful

    print(f"Total Files Analyzed: {len(analysis_results)}")
    print(f"  ✅ Successful: {successful}")
    print(f"  ❌ Failed: {failed}")

    if failed > 0:
        print("\nFailed Files:")
        for r in analysis_results:
            if not r.get("success", False):
                print(f"  - {r['file_path']}: {r.get('error', 'Unknown error')}")

    print_separator("=")
    print("Analysis complete!")

    # Restore stdout and close file
    sys.stdout.close()
    sys.stdout = original_stdout

    # Print confirmation to console
    print(f"✅ Analysis complete! Results saved to: {output_file}")
    print(f"   File size: {output_file.stat().st_size / 1024:.2f} KB")


if __name__ == "__main__":
    main()
