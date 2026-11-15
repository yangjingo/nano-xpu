from nanoxpu.sampling_params import SamplingParams

__all__ = ["LLM", "SamplingParams"]


def __getattr__(name):
    # CLI help and server tests must not load CUDA dependencies.
    if name == "LLM":
        from nanoxpu.llm import LLM
        globals()[name] = LLM
        return LLM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
