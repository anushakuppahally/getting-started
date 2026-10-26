
from inference import predict_robustness


def are_robust(
    model_id: str,
    reasoning_effort: str,
    problems: list[str],
) -> list[bool]:
    return predict_robustness(
        model_id=model_id,
        reasoning_effort=reasoning_effort,
        problems=problems,
    )
