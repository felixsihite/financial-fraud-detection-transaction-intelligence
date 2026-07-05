from src import PROJECT_METADATA, __version__


def test_project_metadata_is_portfolio_ready():
    assert PROJECT_METADATA.name == "Financial Fraud Detection & Transaction Intelligence"
    assert PROJECT_METADATA.version == __version__
    assert PROJECT_METADATA.python_version == "3.13.1"
    assert PROJECT_METADATA.target_column == "isFraud"
    assert PROJECT_METADATA.dataset_source.startswith("https://www.kaggle.com/")