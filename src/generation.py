
"""
src/generation.py - Prompt formatting and baseline generation routines.
"""

import time
from collections import Counter
from typing import Any, Callable

from src.extraction import parse_steps, extract_answer


GenerateFn = Callable[[str, int, dict], list[dict]]


def format_step_prompt(question: str) -> str:
    """
    Create a zero-shot prompt for numbered reasoning steps.

    Returns raw text, without applying a chat template.
    """
    return (
        "Solve the following math word problem carefully.\n"
        "Show your reasoning using numbered steps.\n"
        "Each step must start with 'Step 1:', 'Step 2:', "
        "and so on.\n"
        "After your reasoning, output the final answer on "
        "a separate line using exactly this format:\n"
        "Final Answer: <integer>\n"
        "Do not include any text after the final answer.\n\n"
        f"Question: {question}\n\n"
        "Solution:\n"
    )


def compute_majority_vote(
    answers: list[int | None]
) -> int | None:
    """
    Return the most common valid answer.

    Ignore None values.
    Resolve ties using the earliest occurrence.
    """
    valid_answers = [
        answer for answer in answers
        if answer is not None
    ]

    if not valid_answers:
        return None

    counts = Counter(valid_answers)
    highest_count = max(counts.values())

    # Iterate in original order to resolve ties.
    for answer in valid_answers:
        if counts[answer] == highest_count:
            return answer

    return None


def _generate_with_model(
    model: Any,
    tokenizer: Any,
    prompt: str,
    num_samples: int,
    gen_kwargs: dict
) -> list[dict]:
    """
    Generate text directly with a Hugging Face model.

    Used only when generate_fn is not supplied.
    """
    if model is None or tokenizer is None:
        raise ValueError(
            "Provide model and tokenizer, or a generate_fn."
        )

    import torch

    # Apply the chat template for instruction-tuned models.
    if getattr(tokenizer, "chat_template", None):
        rendered_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        rendered_prompt = prompt

    inputs = tokenizer(
        rendered_prompt,
        return_tensors="pt",
    )

    # Place inputs on the model's input device.
    device = model.get_input_embeddings().weight.device
    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    # Remove None-valued generation arguments.
    generation_kwargs = {
        key: value
        for key, value in gen_kwargs.items()
        if value is not None
    }

    # Ensure the model can generate multiple sequences.
    generation_kwargs["num_return_sequences"] = num_samples

    # Greedy decoding cannot return multiple sequences
    # without beam search.
    if not generation_kwargs.get("do_sample", False):
        if num_samples != 1:
            raise ValueError(
                "Greedy decoding requires num_samples=1."
            )

    if tokenizer.pad_token_id is not None:
        generation_kwargs["pad_token_id"] = (
            tokenizer.pad_token_id
        )
    elif tokenizer.eos_token_id is not None:
        generation_kwargs["pad_token_id"] = (
            tokenizer.eos_token_id
        )

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            **generation_kwargs,
        )

    # Decoder-only models return prompt + completion.
    # Count only newly generated tokens.
    prompt_length = inputs["input_ids"].shape[-1]

    results = []

    for sequence in output_ids:
        new_ids = sequence[prompt_length:]

        # Remove trailing padding from token accounting.
        token_ids = new_ids.tolist()
        pad_id = tokenizer.pad_token_id

        if pad_id is not None:
            while token_ids and token_ids[-1] == pad_id:
                token_ids.pop()

        text = tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
        )

        results.append({
            "text": text,
            "num_tokens": len(token_ids),
        })

    return results


def _call_generator(
    model: Any,
    tokenizer: Any,
    prompt: str,
    num_samples: int,
    gen_kwargs: dict,
    generate_fn: GenerateFn | None,
) -> list[dict]:
    """
    Use a supplied generator or the real Hugging Face model.
    """
    if generate_fn is not None:
        results = generate_fn(
            prompt,
            num_samples,
            gen_kwargs,
        )
    else:
        results = _generate_with_model(
            model,
            tokenizer,
            prompt,
            num_samples,
            gen_kwargs,
        )

    if len(results) != num_samples:
        raise ValueError(
            f"Expected {num_samples} generated samples, "
            f"received {len(results)}."
        )

    return results


def run_greedy(
    model: Any = None,
    tokenizer: Any = None,
    question: str = "",
    generate_fn: GenerateFn | None = None
) -> dict[str, Any]:
    """
    Generate one reasoning path using greedy decoding.

    Returns prompt, completion, steps, answer,
    generation_calls, generated_tokens, time_seconds.
    """
    start_time = time.perf_counter()

    prompt = format_step_prompt(question)

    gen_kwargs = {
        "do_sample": False,
        "temperature": None,
        "top_p": None,
        "top_k": None,
        "repetition_penalty": 1.0,
        "max_new_tokens": 512,
    }

    results = _call_generator(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        num_samples=1,
        gen_kwargs=gen_kwargs,
        generate_fn=generate_fn,
    )

    completion = results[0]["text"]
    num_tokens = int(results[0]["num_tokens"])

    steps = parse_steps(completion)
    answer = extract_answer(completion)

    elapsed = time.perf_counter() - start_time

    return {
        "prompt": prompt,
        "completion": completion,
        "steps": steps,
        "answer": answer,
        "generation_calls": 1,
        "generated_tokens": num_tokens,
        "time_seconds": elapsed,
    }


def run_self_consistency(
    model: Any = None,
    tokenizer: Any = None,
    question: str = "",
    n: int = 5,
    temperature: float = 0.7,
    generate_fn: GenerateFn | None = None
) -> dict[str, Any]:
    """
    Sample n reasoning paths and select the majority answer.

    Returns candidates, answer, generation_calls,
    generated_tokens, time_seconds.
    """
    if n < 1:
        raise ValueError("n must be at least 1.")

    if temperature <= 0:
        raise ValueError(
            "Sampling temperature must be positive."
        )

    start_time = time.perf_counter()

    prompt = format_step_prompt(question)

    gen_kwargs = {
        "do_sample": True,
        "temperature": temperature,
        "top_p": 1.0,
        "top_k": 0,
        "repetition_penalty": 1.0,
        "max_new_tokens": 512,
    }

    results = _call_generator(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        num_samples=n,
        gen_kwargs=gen_kwargs,
        generate_fn=generate_fn,
    )

    candidates = []
    answers = []
    total_tokens = 0

    for result in results:
        completion = result["text"]
        num_tokens = int(result["num_tokens"])

        steps = parse_steps(completion)
        answer = extract_answer(completion)

        candidate = {
            "completion": completion,
            "steps": steps,
            "answer": answer,
            "num_tokens": num_tokens,
        }

        candidates.append(candidate)
        answers.append(answer)
        total_tokens += num_tokens

    majority_answer = compute_majority_vote(answers)

    elapsed = time.perf_counter() - start_time

    return {
        "candidates": candidates,
        "answer": majority_answer,
        "generation_calls": n,
        "generated_tokens": total_tokens,
        "time_seconds": elapsed,
    }
