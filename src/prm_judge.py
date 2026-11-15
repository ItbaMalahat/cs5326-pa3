
"""
src/prm_judge.py - PRM judge prompt construction,
parsing, and step scoring.
"""

import re
from typing import Any, Callable

JudgeFn = Callable[[str], dict]


def build_prm_judge_prompt(
    question: str,
    preceding_steps: list[str],
    current_step: str
) -> str:
    """
    Construct a prompt that asks the judge to evaluate
    one reasoning step under local validity.
    """

    # Format earlier steps in their original order.
    if preceding_steps:
        history = "\n".join(
            f"Step {i}: {step}"
            for i, step in enumerate(preceding_steps, start=1)
        )
    else:
        history = "(no preceding steps)"

    prompt = (
        "You are a mathematical reasoning verifier.\n"
        "Evaluate whether the CURRENT step is logically "
        "and mathematically valid.\n\n"

        "IMPORTANT: Assume all preceding steps are correct.\n"
        "Do not mark the current step invalid solely because "
        "an earlier step contains an error.\n"
        "Evaluate the current step using the problem statement "
        "and the preceding steps as given.\n\n"

        "A step is VALID only if ALL of the following hold:\n"
        "1. Its arithmetic is correct.\n"
        "2. It follows logically from the available information.\n"
        "3. It actually performs a calculation with concrete "
        "numbers, rather than only describing an intention to "
        "calculate something without carrying it out.\n"
        "4. The quantity it computes correctly matches what the "
        "problem is actually asking for or defining at this point. "
        "For example, if the problem asks for a profit, a "
        "difference, or a net change, a step that instead computes "
        "a total, a sum, or an original-plus-increase value is "
        "INVALID even when its arithmetic is correct, because it "
        "answers a different question than the one that was asked.\n"
        "If any of these conditions fails, the step is INVALID.\n\n"

        f"Question:\n{question}\n\n"

        f"Preceding Steps:\n{history}\n\n"

        f"Current Step:\n{current_step}\n\n"

        "Respond using exactly one of these verdict formats:\n"
        "Verdict: VALID\n"
        "Verdict: INVALID\n\n"

        "Then provide one short explanatory sentence:\n"
        "Rationale: <one-sentence explanation>\n\n"

        "Output only one verdict and one rationale."
    )

    return prompt


def parse_judge_decision(
    judge_output: str
) -> tuple[float, str]:
    """
    Convert a judge response into a numerical score
    and a one-sentence rationale.

    VALID   -> +1.0
    INVALID -> -1.0

    Unparseable responses default to INVALID.
    """

    if not isinstance(judge_output, str):
        return -1.0, ""

    # Look for explicit verdict lines first. Not anchored to the
    # start of a line, so markdown like "**Verdict:** VALID" still
    # matches (the ** would otherwise block the ^ anchor).
    # Match INVALID before VALID to avoid substring errors.
    verdict_pattern = re.compile(
        r"Verdict\s*:\s*\**\s*(INVALID|VALID)\b",
        flags=re.IGNORECASE,
    )

    match = verdict_pattern.search(judge_output)

    if match:
        verdict = match.group(1).upper()

        if verdict == "VALID":
            score = 1.0
        else:
            score = -1.0

    else:
        # Fall back to standalone words.
        if re.search(
            r"\bINVALID\b|\bNOT\s+VALID\b",
            judge_output,
            flags=re.IGNORECASE,
        ):
            score = -1.0

        elif re.search(
            r"\bVALID\b",
            judge_output,
            flags=re.IGNORECASE,
        ):
            score = 1.0

        else:
            score = -1.0

    # Find the explanation after "Rationale:".
    rationale_match = re.search(
        r"Rationale\s*:\s*([^\r\n]*)",
        judge_output,
        flags=re.IGNORECASE,
    )

    if not rationale_match:
        rationale = ""

    else:
        rationale_text = rationale_match.group(1).strip()

        # Stop at the first sentence-ending punctuation
        # followed by whitespace or end of text.
        # This does not split decimal numbers like 2.5.
        sentence_match = re.search(
            r"[.!?](?=\s|$)",
            rationale_text,
        )

        if sentence_match:
            rationale = rationale_text[
                :sentence_match.end()
            ].strip()
        else:
            rationale = rationale_text

    return score, rationale


def score_verdict_logprob(
    judge_model: Any,
    judge_tokenizer: Any,
    question: str,
    preceding_steps: list[str],
    current_step: str,
) -> float:
    """
    Optional +3 bonus (PA3.pdf Sec 2.4, item 4): continuous step
    confidence

        p_t = exp(z_VALID) / (exp(z_VALID) + exp(z_INVALID))

    computed from the judge model's raw logits at the verdict token,
    instead of the binary +-1.0 score from parse_judge_decision.

    Approach used: FULL-SEQUENCE LOG-PROBABILITIES, not first-token
    logits. "VALID" and "INVALID" are not guaranteed to tokenize to a
    single, position-aligned token under the Qwen2.5 tokenizer, so a
    first-token comparison could silently misalign. Instead, we teacher-
    force the judge model on the prompt followed by the literal
    continuation "Verdict: VALID" and, separately, "Verdict: INVALID",
    sum each continuation's token-level log-probabilities under the
    model (no sampling, no generation loop - a single forward pass per
    candidate continuation), and softmax the two summed log-probs
    against each other. This reuses an already-generated candidate's
    steps; it does not require any new text generation.

    This report MUST state that the full-sequence approach was used
    (PA3.pdf Sec 2.4 requires stating which of the two approaches was
    chosen).
    """
    import math

    import torch

    prompt = build_prm_judge_prompt(
        question=question,
        preceding_steps=preceding_steps,
        current_step=current_step,
    )
    formatted_prompt = judge_tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_ids = judge_tokenizer(
        formatted_prompt, return_tensors="pt", add_special_tokens=False
    )["input_ids"][0]

    # The two literal continuations we score. Tokenized without special
    # tokens so they concatenate cleanly onto the end of the prompt.
    candidate_texts = {
        "VALID": "Verdict: VALID",
        "INVALID": "Verdict: INVALID",
    }
    summed_logprobs: dict[str, float] = {}

    for label, text in candidate_texts.items():
        cand_ids = judge_tokenizer(
            text, add_special_tokens=False, return_tensors="pt"
        )["input_ids"][0]

        full_ids = torch.cat([prompt_ids, cand_ids]).unsqueeze(0)
        full_ids = full_ids.to(judge_model.device)

        with torch.no_grad():
            logits = judge_model(full_ids).logits[0]

        # logits[i] predicts the token AT position i+1, so the logits
        # that predict the candidate continuation start at
        # (len(prompt_ids) - 1).
        start = prompt_ids.shape[0] - 1
        total_logprob = 0.0

        for j in range(cand_ids.shape[0]):
            step_logits = logits[start + j]
            log_probs = torch.log_softmax(step_logits, dim=-1)
            total_logprob += log_probs[cand_ids[j]].item()

        summed_logprobs[label] = total_logprob

    z_valid = summed_logprobs["VALID"]
    z_invalid = summed_logprobs["INVALID"]

    # Numerically stable 2-way softmax.
    m = max(z_valid, z_invalid)
    p_valid = math.exp(z_valid - m) / (
        math.exp(z_valid - m) + math.exp(z_invalid - m)
    )

    return p_valid


def score_path(
    judge_model: Any = None,
    judge_tokenizer: Any = None,
    question: str = "",
    candidate: dict[str, Any] | None = None,
    judge_fn: JudgeFn | None = None
) -> dict[str, Any]:
    """
    Judge each step sequentially.

    For step t, the judge receives all preceding steps,
    but not any future steps.

    Returns step scores, rationales, judge call count,
    and generated judge token count.
    """

    if candidate is None:
        candidate = {}

    steps = candidate.get("steps", [])

    step_scores = []
    rationales = []
    judge_calls = 0
    judge_tokens = 0

    # No steps means no judge calls.
    if not steps:
        return {
            "step_scores": [],
            "rationales": [],
            "judge_calls": 0,
            "judge_tokens": 0,
        }

    # If no custom judge is supplied, use the model.
    if judge_fn is None:
        if judge_model is None or judge_tokenizer is None:
            raise ValueError(
                "Provide judge_fn or both judge_model "
                "and judge_tokenizer."
            )

        from src.utils import make_hf_judge_fn

        judge_fn = make_hf_judge_fn(
            judge_model,
            judge_tokenizer,
        )

    # Evaluate each step in chronological order.
    for i, current_step in enumerate(steps):

        # All earlier steps, excluding the current step.
        preceding_steps = steps[:i]

        # Build the required verification prompt.
        judge_prompt = build_prm_judge_prompt(
            question=question,
            preceding_steps=preceding_steps,
            current_step=current_step,
        )

        # Call the judge model or mock judge.
        response = judge_fn(judge_prompt)

        # Parse the model's raw response.
        score, rationale = parse_judge_decision(
            response["text"]
        )

        step_scores.append(score)
        rationales.append(rationale)

        # Count one call per judged step.
        judge_calls += 1

        # Count generated tokens, not prompt tokens.
        judge_tokens += int(response["num_tokens"])

    return {
        "step_scores": step_scores,
        "rationales": rationales,
        "judge_calls": judge_calls,
        "judge_tokens": judge_tokens,
    }
