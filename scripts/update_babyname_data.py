from __future__ import annotations

import argparse
import csv
import gzip
import io
import re
import zipfile
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "notebooks" / "data.csv"
DEPLOY_DATA_PATH = PROJECT_ROOT / "data_source" / "app_data.pkl.gz"

YEAR_PATTERN = re.compile(r"yob(?P<year>\d{4})\.txt$")


def parse_birth_totals(html_path: Path) -> dict[int, dict[str, int]]:
    text = html_path.read_text(encoding="utf-8", errors="ignore")
    totals: dict[int, dict[str, int]] = {}
    for match in re.finditer(r"(\d{4})\s+([\d,]+)\s+([\d,]+)\s+([\d,]+)", text):
        year = int(match.group(1))
        male = int(match.group(2).replace(",", ""))
        female = int(match.group(3).replace(",", ""))
        totals[year] = {"M": male, "F": female}
    return totals


def extract_year_records(source_path: Path, target_year: int) -> pd.DataFrame:
    if source_path.suffix.lower() == ".txt":
        year = extract_year_from_name(source_path.name)
        if year != target_year:
            raise ValueError(f"{source_path.name} does not match target year {target_year}.")
        rows = read_year_file(source_path, target_year)
        return pd.DataFrame(rows, columns=["Name", "Gender", "Count", "Year"])

    if source_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(source_path) as archive:
            target_name = f"yob{target_year}.txt"
            if target_name not in archive.namelist():
                raise FileNotFoundError(f"{target_name} was not found inside {source_path.name}.")
            with archive.open(target_name) as handle:
                text = io.TextIOWrapper(handle, encoding="utf-8")
                reader = csv.reader(text)
                rows = [(name, gender, int(count), target_year) for name, gender, count in reader]
        return pd.DataFrame(rows, columns=["Name", "Gender", "Count", "Year"])

    raise ValueError("Source file must be either a yobYYYY.txt file or names.zip.")


def extract_year_from_name(filename: str) -> int:
    match = YEAR_PATTERN.fullmatch(filename)
    if not match:
        raise ValueError(f"Could not infer year from filename: {filename}")
    return int(match.group("year"))


def read_year_file(path: Path, year: int) -> list[tuple[str, str, int, int]]:
    rows: list[tuple[str, str, int, int]] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for name, gender, count in reader:
            rows.append((name, gender, int(count), year))
    return rows


def build_year_frame(year_df: pd.DataFrame, birth_totals: dict[int, dict[str, int]], target_year: int) -> pd.DataFrame:
    totals = birth_totals.get(target_year)
    if not totals:
        raise KeyError(f"Birth totals for {target_year} were not found in the totals file.")

    year_df = year_df.copy()
    year_df["Name_Ratio"] = (
        year_df["Count"] / (totals["M"] + totals["F"]) * 1000
    )
    year_df["Gender_Name_Ratio"] = year_df.apply(
        lambda row: row["Count"] / totals[row["Gender"]] * 1000,
        axis=1,
    )
    return year_df[["Name", "Year", "Gender", "Count", "Name_Ratio", "Gender_Name_Ratio"]]


def refresh_deploy_data(df: pd.DataFrame) -> None:
    deploy_df = df.copy()
    deploy_df["Name"] = deploy_df["Name"].astype("category")
    deploy_df["Gender"] = deploy_df["Gender"].astype("category")
    deploy_df["Year"] = deploy_df["Year"].astype("int16")
    deploy_df["Count"] = deploy_df["Count"].astype("int32")
    deploy_df["Name_Ratio"] = deploy_df["Name_Ratio"].astype("float32")
    deploy_df["Gender_Name_Ratio"] = deploy_df["Gender_Name_Ratio"].astype("float32")
    deploy_df.to_pickle(DEPLOY_DATA_PATH, compression="gzip")


def main() -> None:
    parser = argparse.ArgumentParser(description="Append a new SSA baby name year to the project dataset.")
    parser.add_argument("--source", required=True, help="Path to names.zip or yobYYYY.txt")
    parser.add_argument("--birth-totals-html", required=True, help="Path to saved SSA numberUSbirths.html")
    parser.add_argument("--year", type=int, required=True, help="Target year to import, for example 2024")
    args = parser.parse_args()

    source_path = Path(args.source).expanduser().resolve()
    birth_totals_path = Path(args.birth_totals_html).expanduser().resolve()

    if not source_path.exists():
        raise FileNotFoundError(f"Source file not found: {source_path}")
    if not birth_totals_path.exists():
        raise FileNotFoundError(f"Birth totals file not found: {birth_totals_path}")

    birth_totals = parse_birth_totals(birth_totals_path)
    year_df = extract_year_records(source_path, args.year)
    year_df = build_year_frame(year_df, birth_totals, args.year)

    current_df = pd.read_csv(DATASET_PATH, index_col=0)
    current_df = current_df[current_df["Year"] != args.year]
    updated_df = pd.concat([current_df, year_df], ignore_index=True)
    updated_df = updated_df.sort_values(["Name", "Gender", "Year"]).reset_index(drop=True)

    updated_df.to_csv(DATASET_PATH)
    refresh_deploy_data(updated_df)

    print(f"Updated dataset with {len(year_df):,} rows for {args.year}.")
    print(f"Full dataset now ends at {updated_df['Year'].max()}.")
    print(f"Wrote {DATASET_PATH}")
    print(f"Wrote {DEPLOY_DATA_PATH}")


if __name__ == "__main__":
    main()
