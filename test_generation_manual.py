
from src.generation import (
    format_step_prompt,
    compute_majority_vote,
    run_greedy,
    run_self_consistency,
)


# TEST 1: Prompt formatting
question = "Janet has 16 eggs and uses 7. How many remain?"

prompt = format_step_prompt(question)

assert question in prompt
assert "Step 1:" in prompt
assert "Final Answer:" in prompt

print("TEST 1 PASSED: Prompt formatting")


# TEST 2: Majority voting
assert compute_majority_vote([18, 18, 20, 18, 20]) == 18
assert compute_majority_vote([10, 20, 10, 20]) == 10
assert compute_majority_vote([None, 5, None, 5]) == 5
assert compute_majority_vote([None, None]) is None
assert compute_majority_vote([]) is None
assert compute_majority_vote([-5, 2, -5]) == -5

print("TEST 2 PASSED: Majority voting")


# TEST 3: Mock greedy generation
def mock_greedy(prompt, num_samples, gen_kwargs):
    assert num_samples == 1
    assert gen_kwargs["do_sample"] is False
    assert gen_kwargs["temperature"] is None
    assert gen_kwargs["top_p"] is None
    assert gen_kwargs["top_k"] is None

    return [{
        "text": (
            "Step 1: Janet starts with 16 eggs.\n"
            "Step 2: She uses 7 eggs.\n"
            "Step 3: 16 - 7 = 9.\n"
            "Final Answer: 9"
        ),
        "num_tokens": 40,
    }]


greedy_result = run_greedy(
    question=question,
    generate_fn=mock_greedy,
)

assert greedy_result["answer"] == 9
assert len(greedy_result["steps"]) == 3
assert greedy_result["generation_calls"] == 1
assert greedy_result["generated_tokens"] == 40
assert greedy_result["time_seconds"] >= 0

print("TEST 3 PASSED: Greedy generation")


# TEST 4: Mock self-consistency generation
def mock_self_consistency(prompt, num_samples, gen_kwargs):
    assert num_samples == 5
    assert gen_kwargs["do_sample"] is True
    assert gen_kwargs["temperature"] == 0.7
    assert gen_kwargs["top_p"] == 1.0
    assert gen_kwargs["top_k"] == 0
    assert gen_kwargs["max_new_tokens"] == 512

    answers = [9, 9, 10, 9, 10]

    return [
        {
            "text": (
                f"Step 1: Calculate the remaining eggs.\n"
                f"Final Answer: {answer}"
            ),
            "num_tokens": 20,
        }
        for answer in answers
    ]


sc_result = run_self_consistency(
    question=question,
    n=5,
    temperature=0.7,
    generate_fn=mock_self_consistency,
)

assert sc_result["answer"] == 9
assert len(sc_result["candidates"]) == 5
assert sc_result["generation_calls"] == 5
assert sc_result["generated_tokens"] == 100
assert sc_result["time_seconds"] >= 0

print("TEST 4 PASSED: Self-consistency")


# TEST 5: Candidate structure
for candidate in sc_result["candidates"]:
    assert "completion" in candidate
    assert "steps" in candidate
    assert "answer" in candidate
    assert "num_tokens" in candidate

print("TEST 5 PASSED: Candidate structure")


# TEST 6: Invalid sample count
try:
    run_self_consistency(
        question=question,
        n=0,
        generate_fn=mock_self_consistency,
    )
except ValueError:
    print("TEST 6 PASSED: Invalid sample count")
else:
    raise AssertionError("Expected ValueError for n=0")


print("\nALL GENERATION TESTS PASSED!")
