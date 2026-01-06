# tests/test_task1_preprocessing.py

import pytest
import pandas as pd
from pathlib import Path


@pytest.fixture
def dataset_path():
    """Fixture for dataset path."""
    return Path("data/filtered_complaints.csv")


@pytest.fixture
def df(dataset_path):
    """Fixture to load the dataset."""
    return pd.read_csv(dataset_path)


def test_dataset_exists(dataset_path):
    """Test that the filtered_complaints.csv file exists."""
    assert dataset_path.exists(), f"Dataset not found at {dataset_path}"


def test_dataset_loads_successfully(dataset_path):
    """Test that the CSV loads without errors."""
    try:
        df = pd.read_csv(dataset_path)
        assert df is not None
    except Exception as e:
        pytest.fail(f"Failed to load dataset: {e}")


def test_dataset_not_empty(df):
    """Test that the dataset contains at least one row."""
    assert len(df) > 0, "Dataset is empty"


def test_complaint_narrative_no_nulls(df):
    """Test that consumer_complaint_narrative has no null values."""
    assert df["consumer_complaint_narrative"].isnull().sum() == 0, \
        "Found null values in consumer_complaint_narrative column"


def test_complaint_narrative_no_empty_strings(df):
    """Test that consumer_complaint_narrative has no empty strings."""
    empty_count = (df["consumer_complaint_narrative"].str.strip() == "").sum()
    assert empty_count == 0, \
        f"Found {empty_count} empty strings in consumer_complaint_narrative column"


def test_only_allowed_products(df):
    """Test that only allowed product categories exist."""
    allowed_products = {
        "Credit card",
        "Personal loan",
        "Savings account",
        "Money transfer"
    }
    unique_products = set(df["product"].unique())
    invalid_products = unique_products - allowed_products
    
    assert len(invalid_products) == 0, \
        f"Found invalid products: {invalid_products}. Only allowed: {allowed_products}"


def test_narratives_are_lowercased(df):
    """Test that complaint narratives are lowercased."""
    narratives = df["consumer_complaint_narrative"]
    
    # Check if all narratives are equal to their lowercased version
    non_lowercased = narratives[narratives != narratives.str.lower()]
    
    assert len(non_lowercased) == 0, \
        f"Found {len(non_lowercased)} narratives that are not lowercased"
