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
    assert "test.csv" in os.listdir(directory / "test")


def test_train_test_split(directory):
    train_df = pd.read_csv(directory / "training" / "train.csv", header=None)
    test_df = pd.read_csv(directory / "test" / "test.csv", header=None)

    # no "?"
    assert not (train_df == "?").any().any(), "Found '?' values in training data"
    assert not (test_df == "?").any().any(), "Found '?' values in test data"

    # labels not followed by "."
    assert not any(
        train_df[14].str.contains("<=50K\.") | train_df[14].str.contains(">50K\.")
    )
    assert not any(
        test_df[14].str.contains("<=50K\.") | test_df[14].str.contains(">50K\.")
    )

    # stratified split
    assert (
        train_df[14].value_counts(normalize=True)
        - test_df[14].value_counts(normalize=True)
    ).abs().max() < 0.1

    # mkae sure the label first column
    assert (
        train_df[0].isin(["<=50K", ">50K"]).all()
    ), "Training labels not in expected format"
    assert (
        test_df[0].isin(["<=50K", ">50K"]).all()
    ), "Test labels not in expected format"
