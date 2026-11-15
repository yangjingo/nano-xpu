"""Examples stay readable documents while their code remains executable in CI."""
from tests.helpers import ROOT


def test_examples_are_markdown_and_materialize_all_sources(example_files):
    documents = [path for path in (ROOT / "examples").rglob("*") if path.is_file()]
    assert {path.relative_to(ROOT / "examples").as_posix() for path in documents} == {"server.md", "deploy.md"}
    assert {path.relative_to(example_files).as_posix() for path in example_files.rglob("*") if path.is_file()} == {
        "deployment/cluster-colocated.yaml", "deployment/cluster-pd.yaml",
        "deployment/qwen3-cuda-profile.yaml", "serve/serve-cpu.ps1", "serve/request-cpu.ps1",
        "serve/serve-qwen.sh", "serve/request-qwen.sh",
    }
