import pytest

from tests.helpers.examples import materialize_examples


@pytest.fixture(scope="session")
def example_files(tmp_path_factory):
    return materialize_examples(tmp_path_factory.mktemp("documented-examples"))
