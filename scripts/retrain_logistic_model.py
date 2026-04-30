from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "notebooks" / "data.csv"
FEATURED_DATA_PATH = PROJECT_ROOT / "notebooks" / "Featured_Data.csv"
FAMOUS_NAMES_PATH = PROJECT_ROOT / "data_source" / "famous_names.txt"
MODEL_PATH = PROJECT_ROOT / "models" / "logistic_model.pkl"
PREPROCESSOR_PATH = PROJECT_ROOT / "models" / "preprocessor.pkl"

VOWELS = set("aeiou")
SUPPORTED_ENDINGS = ("a", "e", "i")

BINARY_FEATURES = ["Is_Famous", "Gender_Binary", "Ends_With_Specified_Letters"]
CONTINUOUS_FEATURES = ["Year", "Rolling_Average_Gender_Ratio_5_Years", "Vowel_Count"]
MODEL_FEATURES = BINARY_FEATURES + CONTINUOUS_FEATURES
TARGET_COLUMN = "Is_Top_100"


def load_famous_names(path: Path) -> set[str]:
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def build_featured_dataset(raw_df: pd.DataFrame, famous_names: set[str]) -> pd.DataFrame:
    top_1000 = (
        raw_df.groupby("Year", group_keys=False)
        .apply(lambda frame: frame.nlargest(1000, "Count"))
        .reset_index(drop=True)
    )
    top_1000 = top_1000.sort_values(["Name", "Year", "Gender"]).reset_index(drop=True)

    top_1000["Is_Famous"] = top_1000["Name"].isin(famous_names).astype(int)
    top_1000["Gender_Binary"] = top_1000["Gender"].map({"M": 1, "F": 0}).astype(int)
    top_1000["Year_of_Last_Appearance"] = top_1000.groupby("Name")["Year"].transform("max")

    top_100_per_year = (
        raw_df.groupby("Year", group_keys=False)
        .apply(lambda frame: frame.nlargest(100, "Count"))
        .reset_index(drop=True)
    )
    top_100_marker = top_100_per_year[["Year", "Name"]].drop_duplicates().assign(Is_Top_100=1)
    top_1000 = top_1000.merge(top_100_marker, on=["Year", "Name"], how="left")
    top_1000["Is_Top_100"] = top_1000["Is_Top_100"].fillna(0).astype(int)

    grouped = top_1000.groupby("Name", sort=False)
    top_1000["Rolling_Average_Count_5_Years"] = grouped["Count"].transform(
        lambda series: series.rolling(window=5, min_periods=1).mean()
    )
    top_1000["Rolling_Average_Gender_Ratio_5_Years"] = grouped["Gender_Name_Ratio"].transform(
        lambda series: series.rolling(window=5, min_periods=1).mean()
    )
    top_1000["Rolling_Average_National_Ratio_5_Years"] = grouped["Name_Ratio"].transform(
        lambda series: series.rolling(window=5, min_periods=1).mean()
    )

    top_1000["Yearly_Change_Count"] = grouped["Count"].diff().fillna(0)
    top_1000["Yearly_Change_Gender_Ratio"] = grouped["Gender_Name_Ratio"].diff().fillna(0)
    top_1000["Yearly_Change_National_Ratio"] = grouped["Name_Ratio"].diff().fillna(0)

    lowercase_names = top_1000["Name"].str.lower()
    top_1000["Name_Length"] = top_1000["Name"].str.len()
    top_1000["Vowel_Count"] = lowercase_names.apply(lambda value: sum(char in VOWELS for char in value))
    top_1000["Consonant_Count"] = top_1000["Name_Length"] - top_1000["Vowel_Count"]
    top_1000["Vowel_Ratio"] = top_1000["Vowel_Count"] / top_1000["Name_Length"]
    top_1000["Ends_With_Specified_Letters"] = lowercase_names.str.endswith(SUPPORTED_ENDINGS).astype(int)

    return top_1000


def train_model(featured_df: pd.DataFrame) -> tuple[Pipeline, LogisticRegression, dict[str, float], str]:
    training_df = featured_df[MODEL_FEATURES + [TARGET_COLUMN]].copy()
    training_df["Year"] = training_df["Year"] - 1880

    X = training_df.drop(columns=[TARGET_COLUMN])
    y = training_df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("binary", "passthrough", BINARY_FEATURES),
            (
                "continuous",
                Pipeline(
                    steps=[
                        ("scaler", StandardScaler()),
                        ("transformer", PowerTransformer()),
                    ]
                ),
                CONTINUOUS_FEATURES,
            ),
        ]
    )
    pipeline = Pipeline([("preprocessor", preprocessor)])
    pipeline.fit(X_train)

    X_train_transformed = pipeline.transform(X_train)
    X_test_transformed = pipeline.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_transformed, y_train)

    train_predictions = model.predict(X_train_transformed)
    test_predictions = model.predict(X_test_transformed)
    test_probabilities = model.predict_proba(X_test_transformed)[:, 1]

    metrics = {
        "train_accuracy": accuracy_score(y_train, train_predictions),
        "test_accuracy": accuracy_score(y_test, test_predictions),
        "test_roc_auc": roc_auc_score(y_test, test_probabilities),
    }
    report = classification_report(y_test, test_predictions, digits=4)
    return pipeline, model, metrics, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild the featured dataset and retrain the logistic model.")
    parser.add_argument(
        "--raw-data",
        default=str(RAW_DATA_PATH),
        help="Path to notebooks/data.csv",
    )
    parser.add_argument(
        "--featured-data",
        default=str(FEATURED_DATA_PATH),
        help="Path to notebooks/Featured_Data.csv",
    )
    parser.add_argument(
        "--famous-names",
        default=str(FAMOUS_NAMES_PATH),
        help="Path to a newline-delimited list of famous names",
    )
    args = parser.parse_args()

    raw_data_path = Path(args.raw_data).expanduser().resolve()
    featured_data_path = Path(args.featured_data).expanduser().resolve()
    famous_names_path = Path(args.famous_names).expanduser().resolve()

    raw_df = pd.read_csv(raw_data_path, index_col=0)
    famous_names = load_famous_names(famous_names_path)
    rebuilt_featured_df = build_featured_dataset(raw_df, famous_names)

    pipeline, model, metrics, report = train_model(rebuilt_featured_df)

    featured_data_path.parent.mkdir(parents=True, exist_ok=True)
    rebuilt_featured_df.to_csv(featured_data_path, index=False)
    joblib.dump(pipeline, PREPROCESSOR_PATH)
    joblib.dump(model, MODEL_PATH)

    print(
        "Rebuilt Featured_Data.csv with "
        f"{len(rebuilt_featured_df):,} rows spanning {rebuilt_featured_df['Year'].min()}-"
        f"{rebuilt_featured_df['Year'].max()}."
    )
    print(f"Saved {PREPROCESSOR_PATH}")
    print(f"Saved {MODEL_PATH}")
    print(f"Train accuracy: {metrics['train_accuracy']:.4f}")
    print(f"Test accuracy: {metrics['test_accuracy']:.4f}")
    print(f"Test ROC AUC: {metrics['test_roc_auc']:.4f}")
    print("Classification report:")
    print(report)


if __name__ == "__main__":
    main()
