
from src.prm_judge import (
    build_prm_judge_prompt,
    parse_judge_decision,
    score_path,
)


# TEST 1: Judge prompt formatting
prompt = build_prm_judge_prompt(
    "What is 2 + 2?",
    [],
    "2 + 2 = 4",
)

assert "(no preceding steps)" in prompt
assert "Current Step:" in prompt
assert "Verdict: VALID" in prompt
assert "Verdict: INVALID" in prompt
assert "Rationale:" in prompt
assert "assume all preceding steps are correct" in prompt.lower()

print("TEST 1 PASSED: Judge prompt formatting")


# TEST 2: Preceding steps
prompt = build_prm_judge_prompt(
    "What is 2 + 2?",
    ["2 + 2 = 4", "4 * 2 = 8"],
    "8 + 1 = 9",
)

assert "Step 1: 2 + 2 = 4" in prompt
assert "Step 2: 4 * 2 = 8" in prompt

print("TEST 2 PASSED: Preceding steps")


# TEST 3: Judge response parsing
cases = [
    (
        "Verdict: VALID\nRationale: Correct arithmetic.",
        1.0,
        "Correct arithmetic.",
    ),
    (
        "Verdict: INVALID\nRationale: Wrong subtraction.",
        -1.0,
        "Wrong subtraction.",
    ),
    (
        "Verdict: **VALID**\nRationale: Correct algebra.",
        1.0,
        "Correct algebra.",
    ),
    (
        "The step is NOT VALID.",
        -1.0,
        "",
    ),
    (
        "The step is invalid.",
        -1.0,
        "",
    ),
    (
        "Verdict: VALID\nRationale: 2.5 * 4 = 10 is correct. Extra text.",
        1.0,
        "2.5 * 4 = 10 is correct.",
    ),
    (
        "Total gibberish.",
        -1.0,
        "",
    ),
]

for text, expected_score, expected_rationale in cases:
    score, rationale = parse_judge_decision(text)

    assert score == expected_score, (
        f"Wrong score for {text!r}"
    )
    assert rationale == expected_rationale, (
        f"Wrong rationale for {text!r}: {rationale!r}"
    )

print("TEST 3 PASSED: Judge response parsing")


# TEST 4: Sequential step scoring
responses = iter([
    {
        "text": "Verdict: VALID\nRationale: Correct start.",
        "num_tokens": 8,
    },
    {
        "text": "Verdict: INVALID\nRationale: Arithmetic error.",
        "num_tokens": 10,
    },
    {
        "text": "Verdict: VALID\nRationale: Valid given previous steps.",
        "num_tokens": 12,
    },
])

captured_prompts = []

def mock_judge(prompt):
    captured_prompts.append(prompt)
    return next(responses)


candidate = {
    "steps": [
        "Janet has 16 eggs.",
        "16 - 7 = 8",
        "8 * 2 = 16",
    ]
}

result = score_path(
    question="Janet has 16 eggs and uses 7.",
    candidate=candidate,
    judge_fn=mock_judge,
)

assert result["step_scores"] == [1.0, -1.0, 1.0]
assert result["judge_calls"] == 3
assert result["judge_tokens"] == 30
assert len(result["rationales"]) == 3

# The third judge prompt must contain the first
# two steps but not any future steps.
assert "Step 1: Janet has 16 eggs." in captured_prompts[2]
assert "Step 2: 16 - 7 = 8" in captured_prompts[2]

print("TEST 4 PASSED: Sequential step scoring")


# TEST 5: Empty reasoning path
empty_result = score_path(
    question="What is 2 + 2?",
    candidate={"steps": []},
    judge_fn=mock_judge,
)

assert empty_result["step_scores"] == []
assert empty_result["rationales"] == []
assert empty_result["judge_calls"] == 0
assert empty_result["judge_tokens"] == 0

print("TEST 5 PASSED: Empty reasoning path")


print("\nALL PRM TESTS PASSED!")
