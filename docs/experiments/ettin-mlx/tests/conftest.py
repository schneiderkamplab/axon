import json

import pytest

from helpers import DATA

REPORT = DATA / "parity_report.json"


class Report:
    """Collects measured numbers across the session into tests/data/parity_report.json."""

    def __init__(self):
        self.data = json.loads(REPORT.read_text()) if REPORT.exists() else {}

    def add(self, model_name: str, key: str, values: dict):
        self.data.setdefault(model_name, {})[key] = values

    def save(self):
        DATA.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(json.dumps(self.data, indent=1, sort_keys=True))


@pytest.fixture(scope="session")
def report():
    r = Report()
    yield r
    r.save()
