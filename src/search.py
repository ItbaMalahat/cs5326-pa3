
"""
src/search.py - Path aggregation, Best-of-N search,
and compute-matched search.
"""

import time
from typing import Any, Callable

from src.generation import (
    compute_majority_vote,
    run_self_consistency,
)
from src.prm_judge import score_path


GenerateFn = Callable[[str, int, dict], list[dict]]
JudgeFn = Callable[[str], dict]


def aggregate_path_score(
    step_scores: list[float],
    strategy: str = "min",
    penalty_lambda: float = 2.0
) -> float:
    """
    Combine individual PRM step scores into one path score.

    Supported strategies:
      min
      product
      step_penalty
    """

    # Reject unsupported strategies.
    if strategy not in {"min", "product", "step_penalty"}:
        raise ValueError(
            f"Unknown aggregation strategy: {strategy}"
        )

    # A path without reasoning steps gets the lowest
    # default score, regardless of aggregation strategy.
    if not step_scores:
        return -1.0

    if strategy == "min":
        # A single invalid step makes the minimum -1.
        return float(min(step_scores))

    if strategy == "product":
        product = 1.0

        for score in step_scores:
            # Convert score from [-1, 1] to [0, 1].
            probability = (score + 1.0) / 2.0

            # Avoid exact zero or one.
            probability = max(
                0.01,
                min(0.99, probability),
            )

            product *= probability

        return float(product)

    # strategy == "step_penalty"

    mean_score = sum(step_scores) / len(step_scores)

    invalid_count = sum(
        1 for score in step_scores if score == -1.0
    )

    return float(
        mean_score - penalty_lambda * invalid_count
    )


def select_best_path(
    candidate_paths: list[dict[str, Any]],
    aggregation_strategy: str = "min",
    penalty_lambda: float = 2.0
) -> dict[str, Any]:
    """
    Select the highest-scoring eligible candidate.

    If scores tie, prefer the majority answer among
    tied candidates, then the earliest candidate.

    Return a shallow copy without modifying inputs.
    """

    if not candidate_paths:
        raise ValueError("No candidate paths provided.")

    # Compute scores separately to avoid modifying
    # the original candidate dictionaries.
    path_scores = [
        aggregate_path_score(
            candidate["step_scores"],
            strategy=aggregation_strategy,
            penalty_lambda=penalty_lambda,
        )
        for candidate in candidate_paths
    ]

    # Candidates with an extracted numerical answer
    # are preferred over candidates with answer=None.
    valid_indices = [
        i for i, candidate in enumerate(candidate_paths)
        if candidate["answer"] is not None
    ]

    # If every answer is None, consider all candidates.
    eligible_indices = (
        valid_indices
        if valid_indices
        else list(range(len(candidate_paths)))
    )

    # Find the highest score among eligible candidates.
    best_score = max(
        path_scores[i] for i in eligible_indices
    )

    # Treat scores within 1e-9 as tied.
    tolerance = 1e-9

    tied_indices = [
        i for i in eligible_indices
        if abs(path_scores[i] - best_score) <= tolerance
    ]

    if len(tied_indices) == 1:
        selected_index = tied_indices[0]

    else:
        # Vote using ONLY the tied candidates.
        tied_answers = [
            candidate_paths[i]["answer"]
            for i in tied_indices
        ]

        majority_answer = compute_majority_vote(
            tied_answers
        )

        if majority_answer is None:
            # All tied answers are None.
            selected_index = tied_indices[0]
        else:
            # The first tied candidate with the
            # majority answer wins.
            selected_index = next(
                i for i in tied_indices
                if candidate_paths[i]["answer"]
                == majority_answer
            )

    # Shallow copy preserves all original fields.
    selected = candidate_paths[selected_index].copy()

    selected["path_score"] = float(
        path_scores[selected_index]
    )
    selected["num_tied"] = len(tied_indices)

    return selected


def run_prm_best_of_n(
    model: Any = None,
    judge_model: Any = None,
    gen_tokenizer: Any = None,
    judge_tokenizer: Any = None,
    question: str = "",
    n: int = 5,
    aggregation_strategy: str = "min",
    penalty_lambda: float = 2.0,
    candidates: list[dict[str, Any]] | None = None,
    generate_fn: GenerateFn | None = None,
    judge_fn: JudgeFn | None = None
) -> dict[str, Any]:
    """
    Generate or reuse candidate solutions, score each
    with the PRM judge, and select the best path.
    """

    start_time = time.perf_counter()

    # CASE 1: No existing candidates.
    # Generate n solutions using self-consistency.
    if candidates is None:
        generation_result = run_self_consistency(
            model=model,
            tokenizer=gen_tokenizer,
            question=question,
            n=n,
            temperature=0.7,
            generate_fn=generate_fn,
        )

        candidate_pool = generation_result["candidates"]

        generation_calls = generation_result[
            "generation_calls"
        ]
        generated_tokens = generation_result[
            "generated_tokens"
        ]

    else:
        # CASE 2: Reuse candidates.
        # No additional generation is allowed.
        candidate_pool = candidates
        generation_calls = 0
        generated_tokens = 0

    scored_candidates = []

    total_judge_calls = 0
    total_judge_tokens = 0

    for candidate in candidate_pool:

        # Judge every reasoning step in this candidate.
        scoring_result = score_path(
            judge_model=judge_model,
            judge_tokenizer=judge_tokenizer,
            question=question,
            candidate=candidate,
            judge_fn=judge_fn,
        )

        # Make a copy so the shared candidate pool
        # remains unchanged.
        scored_candidate = candidate.copy()

        scored_candidate["step_scores"] = scoring_result[
            "step_scores"
        ]
        scored_candidate["rationales"] = scoring_result[
            "rationales"
        ]

        scored_candidates.append(scored_candidate)

        total_judge_calls += scoring_result[
            "judge_calls"
        ]
        total_judge_tokens += scoring_result[
            "judge_tokens"
        ]

    # Choose the best candidate using the requested
    # aggregation strategy.
    selected = select_best_path(
        candidate_paths=scored_candidates,
        aggregation_strategy=aggregation_strategy,
        penalty_lambda=penalty_lambda,
    )

    elapsed = time.perf_counter() - start_time

    return {
        "selected": selected,
        "answer": selected["answer"],
        "candidates": scored_candidates,
        "generation_calls": generation_calls,
        "judge_calls": total_judge_calls,
        "generated_tokens": generated_tokens,
        "judge_tokens": total_judge_tokens,
        "time_seconds": elapsed,
    }


def run_compute_matched_self_consistency(
    model: Any = None,
    tokenizer: Any = None,
    question: str = "",
    target_calls: int = 15,
    temperature: float = 0.7,
    existing_candidates: list[dict[str, Any]] | None = None,
    generate_fn: GenerateFn | None = None
) -> dict[str, Any]:
    """
    Extend an existing candidate pool to target_calls
    total samples, then apply majority voting.

    Generation statistics count ONLY new samples.
    """

    if target_calls < 1:
        raise ValueError(
            "target_calls must be at least 1."
        )

    start_time = time.perf_counter()

    # Copy the list so the caller's list is unchanged.
    all_candidates = (
        list(existing_candidates)
        if existing_candidates is not None
        else []
    )

    if len(all_candidates) > target_calls:
        raise ValueError(
            "Existing candidates exceed target_calls."
        )

    # Determine how many additional samples are needed.
    additional_samples = (
        target_calls - len(all_candidates)
    )

    generation_calls = 0
    generated_tokens = 0

    if additional_samples > 0:

        # Generate ONLY the missing samples.
        new_result = run_self_consistency(
            model=model,
            tokenizer=tokenizer,
            question=question,
            n=additional_samples,
            temperature=temperature,
            generate_fn=generate_fn,
        )

        # Preserve original candidates first,
        # followed by newly generated candidates.
        all_candidates.extend(
            new_result["candidates"]
        )

        generation_calls = new_result[
            "generation_calls"
        ]
        generated_tokens = new_result[
            "generated_tokens"
        ]

    # Majority voting across ALL candidate answers.
    answers = [
        candidate["answer"]
        for candidate in all_candidates
    ]

    final_answer = compute_majority_vote(answers)

    elapsed = time.perf_counter() - start_time

    return {
        "candidates": all_candidates,
        "answer": final_answer,
        "generation_calls": generation_calls,
        "total_samples": len(all_candidates),
        "generated_tokens": generated_tokens,
        "time_seconds": elapsed,
    }
