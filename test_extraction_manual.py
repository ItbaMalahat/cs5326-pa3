
from src.extraction import parse_steps, extract_answer

# TEST 1: Normal numbered steps
text = """
Step 1: Janet has 16 eggs.
Step 2: She uses 7 eggs.
Step 3: She has 9 eggs remaining.
Final Answer: 18
"""

steps = parse_steps(text)

assert len(steps) == 3
assert steps[0] == "Janet has 16 eggs."
assert steps[1] == "She uses 7 eggs."
assert steps[2] == "She has 9 eggs remaining."

print("TEST 1 PASSED: Normal steps")


# TEST 2: Markdown-formatted steps
text = """
**Step 1:** Calculate the total.
### Step 2: Subtract the used eggs.
**Final Answer:** 18
"""

steps = parse_steps(text)

assert len(steps) == 2
assert steps[0] == "Calculate the total."
assert steps[1] == "Subtract the used eggs."

print("TEST 2 PASSED: Markdown steps")


# TEST 3: Answer extraction
test_cases = [
    ("Final Answer: 18", 18),
    ("Final Answer: $1,234", 1234),
    ("Final Answer: 18.00", 18),
    ("Final Answer: 18.5", None),
    ("Final Answer: -5", -5),
    ("Final Answer: 18 dollars.", 18),
    ("**Final Answer:** 18", 18),
    ("final answer 18", 18),
    ("Final Answer: 10\nFinal Answer: 20", 20),
    ("The answer is 18", None),
    ("", None),
    (None, None),
]

for text, expected in test_cases:
    actual = extract_answer(text)

    assert actual == expected, (
        f"Failed for {text!r}: "
        f"expected {expected}, got {actual}"
    )

print("TEST 3 PASSED: Answer extraction")


# TEST 4: Missing steps
assert parse_steps("") == []
assert parse_steps(None) == []
assert parse_steps("Final Answer: 18") == []

print("TEST 4 PASSED: Missing steps")


print("\nALL EXTRACTION TESTS PASSED!")
