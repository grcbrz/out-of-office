from __future__ import annotations

from src.monitoring.drift.evidently_report import EvidentlyReporter


def test_evidently_report_frequency():
    reporter = EvidentlyReporter(output_dir="/tmp", report_frequency_days=7)
    assert not reporter.should_generate(1)
    assert not reporter.should_generate(6)
    assert reporter.should_generate(7)
    assert not reporter.should_generate(8)
    assert reporter.should_generate(14)
