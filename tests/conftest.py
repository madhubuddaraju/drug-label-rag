from pathlib import Path

import pytest

from labelrag import corpus

FIXTURE = Path(__file__).parent / "fixtures" / "labels_sample.jsonl.gz"


@pytest.fixture(scope="session")
def labels():
    return corpus.load_labels(FIXTURE)


@pytest.fixture(scope="session")
def chunks(labels):
    return corpus.build_chunks(labels)


@pytest.fixture(scope="session")
def aliases(labels):
    return corpus.alias_index(labels)
