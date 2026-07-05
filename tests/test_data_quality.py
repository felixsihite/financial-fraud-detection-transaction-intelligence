from src.data.data_quality import compute_quality_report


def test_compute_quality_report_on_small_csv(sample_transactions, tmp_path):
    input_path = tmp_path / "transactions.csv"
    output_dir = tmp_path / "reports"
    sample_transactions.to_csv(input_path, index=False)

    report = compute_quality_report(input_path, output_dir=output_dir, chunksize=2)

    assert report["row_count"] == 4
    assert report["column_count"] == 11
    assert report["total_missing_values"] == 0
    assert report["target_distribution"]["fraud"] == 2
    assert report["expected_columns_match"] is True
    assert report["flagged_fraud_capture_rate"] == 0
    assert report["leakage_audit"]["target_column_excluded_from_features"] is True
    assert (output_dir / "data_quality_summary.json").exists()
    assert (output_dir / "data_quality_summary.md").exists()


def test_quality_markdown_contains_portfolio_sections(sample_transactions, tmp_path):
    input_path = tmp_path / "transactions.csv"
    output_dir = tmp_path / "reports"
    sample_transactions.to_csv(input_path, index=False)

    compute_quality_report(input_path, output_dir=output_dir, chunksize=2)
    markdown = (output_dir / "data_quality_summary.md").read_text(encoding="utf-8")

    assert "Data Quality and Leakage Audit" in markdown
    assert "Transaction Type Fraud Distribution" in markdown
    assert "Legacy Rule Audit" in markdown
    assert "Dataset Limitations" in markdown