
"""
src/extraction.py - Step and answer extraction utilities.
"""

import re


def parse_steps(completion_text: str) -> list[str]:
    """
    Extract numbered reasoning steps from a model completion.

    Supports:
      Step 1: ...
      Step 2. ...
      **Step 1:** ...
      ### Step 2: ...
      `Step 3:` ...

    Returns only step contents, without labels or Markdown.
    """
    if not isinstance(completion_text, str) or not completion_text:
        return []

    # Match a step label at the beginning of a line.
    # Markdown decoration around the label is optional.
    step_pattern = re.compile(
        r"^[ \t]*[#*` \t]*Step[ \t]+\d+[ \t]*[:.]?"
        r"[*` \t]*",
        flags=re.IGNORECASE | re.MULTILINE,
    )

    # Identify the final-answer label so it is not
    # accidentally included in the last reasoning step.
    final_pattern = re.compile(
        r"^[ \t]*[#*` \t]*Final[ \t]+Answer"
        r"[ \t]*[:]?[*` \t]*",
        flags=re.IGNORECASE | re.MULTILINE,
    )

    step_matches = list(step_pattern.finditer(completion_text))
    final_matches = list(final_pattern.finditer(completion_text))

    steps = []

    for i, match in enumerate(step_matches):
        start = match.end()

        # By default, the step ends at the end of the text.
        end = len(completion_text)

        # Stop before the next numbered step.
        if i + 1 < len(step_matches):
            end = step_matches[i + 1].start()

        # Stop before a final-answer label if it
        # appears within this step's boundaries.
        for final_match in final_matches:
            if start <= final_match.start() < end:
                end = final_match.start()
                break

        step_text = completion_text[start:end]

        # Remove Markdown decoration left around the text.
        step_text = step_text.strip().strip("*#`").strip()

        if step_text:
            steps.append(step_text)

    return steps


def extract_answer(completion_text: str) -> int | None:
    """
    Extract the integer following the LAST Final Answer label.

    Examples:
      Final Answer: 18          -> 18
      Final Answer: $1,234      -> 1234
      Final Answer: 18.00       -> 18
      Final Answer: 18.5        -> None
      Final Answer: -5          -> -5
      Final Answer: 18 dollars. -> 18
      **Final Answer:** 18      -> 18

    Returns None if the answer is missing or non-integer.
    """
    if not isinstance(completion_text, str) or not completion_text:
        return None

    # Match final-answer labels, including Markdown.
    label_pattern = re.compile(
        r"^[ \t]*[#*` \t]*Final[ \t]+Answer"
        r"[ \t]*[:]?[*` \t]*",
        flags=re.IGNORECASE | re.MULTILINE,
    )

    matches = list(label_pattern.finditer(completion_text))

    # Never guess an answer if the required label is absent.
    if not matches:
        return None

    # The assignment requires the LAST final-answer label.
    last_match = matches[-1]

    # Read the content on the same line as the label.
    answer_line = completion_text[
        last_match.end():
    ].split("\n", 1)[0].strip()

    # Allow an optional currency symbol, sign, commas,
    # and a decimal component.
    number_match = re.match(
        r"^\$?\s*([+-]?\d[\d,]*(?:\.\d+)?)"
        r"(?![\d.,])",
        answer_line,
    )

    if not number_match:
        return None

    number_text = number_match.group(1).replace(",", "")

    # Decimal values are accepted only when mathematically
    # equal to an integer (e.g. 18.00).
    if "." in number_text:
        integer_part, fractional_part = number_text.split(".", 1)

        if not fractional_part or any(
            digit != "0" for digit in fractional_part
        ):
            return None

        return int(integer_part)

    return int(number_text)
