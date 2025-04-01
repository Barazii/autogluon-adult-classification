import pytest
import shutil
import tempfile
from pathlib import Path
import os
from src.processing import processing
import subprocess
import dotenv
import pandas as pd

dotenv.load_dotenv()


@pytest.fixture(scope="function", autouse=False)
def directory():
    directory = Path(tempfile.mkdtemp())
    data_directory = directory / "rawdata"
    data_directory.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            "aws",
            "s3",
            "cp",
            os.environ["S3_DATA_URI"],
            str(data_directory),
            "--recursive",
            "--quiet",
        ],
        check=True,
    )

    processing(directory)

    yield directory

    shutil.rmtree(directory)


def test_directories_contents(directory):
    output_directories = os.listdir(directory)
    assert "training" in output_directories
    assert "test" in output_directories
    assert "train.csv" in os.listdir(directory / "training")
    assert "test_without_labels.csv" in os.listdir(directory / "test")
    assert "test_labels.csv" in os.listdir(directory / "test")
    assert "test_with_labels.csv" in os.listdir(directory / "test")


def test_train_test_split(directory):
    train_df = pd.read_csv(directory / "training" / "train.csv", header=None)
    test_with_labels_df = pd.read_csv(directory / "test" / "test_with_labels.csv", header=None)
    test_without_labels_df = pd.read_csv(directory / "test" / "test_without_labels.csv", header=None)
    test_labels_df = pd.read_csv(directory / "test" / "test_labels.csv", header=None)

    # no "?"
    assert not (train_df == "?").any().any(), "Found '?' values in training data"
    assert not (test_with_labels_df == "?").any().any(), "Found '?' values in test data"

    # find column with income labels and make sure it's first column
    income_col_train = [
        i for i, col in train_df.items() if col.isin(["<=50K", ">50K"]).all()
    ][0]
    income_col_test = [
        i for i, col in test_with_labels_df.items() if col.isin(["<=50K", ">50K"]).all()
    ][0]
    income_col_train == 0, "Income column not first in training data"
    income_col_test == 14

    income_col_test_without_labels = [
        i for i, col in test_without_labels_df.items() if col.isin(["<=50K", ">50K"]).all()
    ]
    assert len(income_col_test_without_labels) == 0, "Income column found in test data"

    # labels not followed by "."
    assert not any(
        train_df[income_col_train].str.contains("<=50K\.")
        | train_df[income_col_train].str.contains(">50K\.")
    )
    assert not any(
        test_labels_df[0].str.contains("<=50K\.")
        | test_labels_df[0].str.contains(">50K\.")
    )

    # stratified split
    assert (
        train_df[income_col_train].value_counts(normalize=True)
        - test_with_labels_df[income_col_test].value_counts(normalize=True)
    ).abs().max() < 0.01, "There is imbalance between the train data and test data"
