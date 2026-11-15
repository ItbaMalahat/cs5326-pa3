
import copy
import math

from src.search import (
    aggregate_path_score,
    select_best_path,
    run_prm_best_of_n,
    run_compute_matched_self_consistency,
)


# TEST 1: Aggregation strategies
scores = [1.0, -1.0, 1.0]

assert aggregate_path_score(scores, "min") == -1.0

assert math.isclose(
    aggregate_path_score(scores, "product"),
    0.009801,
)

assert math.isclose(
    aggregate_path_score(scores, "step_penalty"),
    -1.6666666666666667,
)

for strategy in ["min", "product", "step_penalty"]:
    assert aggregate_path_score([], strategy) == -1.0

try:
    aggregate_path_score([1.0], "unknown")
except ValueError:
    pass
else:
    raise AssertionError("Unknown strategy must raise ValueError")

print("TEST 1 PASSED: Aggregation strategies")


# TEST 2: Clear winner
candidates = [
    {"step_scores": [-1.0], "answer": 10},
    {"step_scores": [1.0], "answer": 20},
]

original = copy.deepcopy(candidates)

winner = select_best_path(candidates)

assert winner["answer"] == 20
assert winner["path_score"] == 1.0
assert winner["num_tied"] == 1
assert candidates == original
assert winner is not candidates[1]

print("TEST 2 PASSED: Clear winner and no mutation")


# TEST 3: Tie-breaking by majority
candidates = [
    {"step_scores": [1.0], "answer": 10},
    {"step_scores": [1.0], "answer": 20},
    {"step_scores": [1.0], "answer": 20},
]

winner = select_best_path(candidates)

assert winner["answer"] == 20
assert winner["num_tied"] == 3

print("TEST 3 PASSED: Majority tie-breaking")


# TEST 4: Missing answers
candidates = [
    {"step_scores": [1.0], "answer": None},
    {"step_scores": [-1.0], "answer": 42},
]

winner = select_best_path(candidates)

assert winner["answer"] == 42

print("TEST 4 PASSED: Missing answer exclusion")


# TEST 5: PRM Best-of-N using shared candidates
shared_candidates = [
    {
        "completion": "Step 1: Wrong.\nFinal Answer: 10",
        "steps": ["Wrong."],
        "answer": 10,
        "num_tokens": 15,
    },
    {
        "completion": "Step 1: Correct.\nFinal Answer: 20",
        "steps": ["Correct."],
        "answer": 20,
        "num_tokens": 15,
    },
]

original_shared = copy.deepcopy(shared_candidates)

judge_responses = iter([
    {
        "text": "Verdict: INVALID\nRationale: Math error.",
        "num_tokens": 10,
    },
    {
        "text": "Verdict: VALID\nRationale: Correct.",
        "num_tokens": 12,
    },
])


def mock_judge(prompt):
    return next(judge_responses)


def forbidden_generator(prompt, n, kwargs):
    raise AssertionError(
        "Generator must not be called for shared candidates."
    )


result = run_prm_best_of_n(
    question="Test problem",
    n=2,
    candidates=shared_candidates,
    generate_fn=forbidden_generator,
    judge_fn=mock_judge,
)

assert result["answer"] == 20
assert result["generation_calls"] == 0
assert result["generated_tokens"] == 0
assert result["judge_calls"] == 2
assert result["judge_tokens"] == 22
assert shared_candidates == original_shared

print("TEST 5 PASSED: PRM Best-of-N with shared candidates")


# TEST 6: Compute-matched Self-Consistency
existing = [
    {
        "completion": "Existing A",
        "steps": ["A"],
        "answer": 10,
        "num_tokens": 10,
    },
    {
        "completion": "Existing B",
        "steps": ["B"],
        "answer": 20,
        "num_tokens": 10,
    },
]


def mock_generator(prompt, n, kwargs):
    assert n == 3
    assert kwargs["do_sample"] is True

    return [
        {
            "text": f"Step 1: Work.\nFinal Answer: {answer}",
            "num_tokens": 12,
        }
        for answer in [20, 20, 30]
    ]


result = run_compute_matched_self_consistency(
    question="Test problem",
    target_calls=5,
    existing_candidates=existing,
    generate_fn=mock_generator,
)

assert result["answer"] == 20
assert result["total_samples"] == 5
assert result["generation_calls"] == 3
assert result["generated_tokens"] == 36
assert len(result["candidates"]) == 5
assert result["candidates"][:2] == existing

print("TEST 6 PASSED: Compute-matched Self-Consistency")


# TEST 7: No extra generation if target already reached
result = run_compute_matched_self_consistency(
    question="Test problem",
    target_calls=2,
    existing_candidates=existing,
    generate_fn=forbidden_generator,
)

assert result["generation_calls"] == 0
assert result["generated_tokens"] == 0
assert result["total_samples"] == 2
assert result["answer"] == 10

print("TEST 7 PASSED: No unnecessary generation")


print("\nALL SEARCH TESTS PASSED!")
