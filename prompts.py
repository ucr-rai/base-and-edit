prompt_lean2nl = r"""
Task:
You are given a formal mathematical proof written as a Lean4 code, provided in a "<proof>" tag.
Translate it into an informal proof written in natural language.

<proof>
{proof}
</proof>

Instructions:
1. If the formal proof contains several steps, try to do the translation step by step.
2. Make sure that the reasoning in your translated proof is self-contained and matches the formal proof, step by step.
3. The provided formal proof may contain comments which you may leverage.
However, the existing comments alone may not be complete,
and thus you should consider both comments and other parts of the code in the formal proof.
4. Directly give the translated result. Do not output any thing extra.
"""

prompt_lean2nl_without_comments = r"""
Task:
You are given a formal mathematical proof written as a Lean4 code, provided in a "<proof>" tag.
Translate it into an informal proof written in natural language.

<proof>
{proof}
</proof>

Instructions:
1. If the formal proof contains several steps, try to do the translation step by step.
2. Make sure that the reasoning in your translated proof is self-contained and matches the formal proof, step by step.
3. Directly give the translated result. Do not output any thing extra.
"""

prompt_herald = r"""
<informal_theorem>
{informal_theorem}
</informal_theorem>

<formal_theorem>
{formal_theorem}
</formal_theorem>

<informal_proof>
{informal_proof}
</informal_proof>
"""

prompt_informal2formal = r"""Given an informal proof written in natural language, translate it into a formal proof in Lean4.

<informal_proof>
{informal_proof}
</informal_proof>
"""

prompt_informal2formal_with_problem = r"""For a math proof problem (provided in the "<problem>" tag), given an informal proof written in natural language (provided in the "<proof>" tag), translate it into a formal proof in Lean4.

<problem>
{problem}
</problem>

<proof>
{proof}
</proof>
"""

prompt_informal_problem_only = r"""For a math proof problem (provided in the "<problem>" tag), write a formal proof in Lean4.

<problem>
{problem}
</problem>
"""

# Removed '\\n<|EOT|>\\n' and "{{'### Response:\\n'}}\n"
# chat_template_partial = "{%- set found_item = false -%}\n{%- for message in messages -%}\n    {%- if message['role'] == 'system' -%}\n        {%- set found_item = true -%}\n    {%- endif -%}\n{%- endfor -%}\n{%- if not found_item -%}\n{{'You are a helpful AI assistant.\\n'}}\n{%- endif %}\n{%- for message in messages %}\n    {%- if message['role'] == 'system' %}\n{{ message['content'] }}\n    {%- else %}\n        {%- if message['role'] == 'user' %}\n{{'### Instruction:\\n' + message['content'] + '\\n'}}\n        {%- else %}\n{{'### Response:\\n' + message['content']}}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}"
chat_template_partial = "{% for message in messages %}{%- if message['role'] == 'user' %}{{'<|im_start|>user\n' + message['content'] + '<|im_end|>' + '\n'}}{%- else %}{{'<|im_start|>assistant\n' + message['content']}}{%- endif %}{% endfor %}{% if end_generation %}{{ '<|im_end|>' + '\n' }}{% endif %}"

prompt_solve_informal = r"""
Solve the given math problem and provide your reasoning step-by-step.

{informal_problem}
"""

prompt_solve_informal_gpt41 = r"""
Solve the given math problem.
Please provide a proof step by step.
Do not output your thinking that is not an essential part of the proof.
Do not include steps that you eventually discard.

{informal_problem}
"""


prompt_translate_statement = """
Given a problem written in natural language, translate it into a formal statement in Lean4.

The problem written in the informal natural language is in the "<informal_problem>" tag.

Your output should be a Lean4 code which is the formal statement (followed by “by sorry”) that can pass Lean4 compilation.

Directly output the code without anything else. Begin your output by "theorem" (no need to add headers).

Please refer to examples below.
In the formal statement, you should declare any variables as parameters, not as ∀ quantifiers.
Your formal statement should meaningfully cover the informal problem, instead of simply putting a final result.

Good: theorem sample_theorem (a b c : ℝ) (h : a / (b + c) = 1) : a ^ 2 + b ^ 2 + c ^ 2 ≥ (6 / 5) * (a * b + b * c + c * a) := by sorry
Bad: theorem sample_theorem : ∀ (a b c : ℝ), a / (b + c) = 1 → a ^ 2 + b ^ 2 + c ^ 2 ≥ (6 / 5) * (a * b + b * c + c * a) := by sorry
Bad: theorem sample_theorem : True := by sorry
Bad: theorem sample_theorem : 1 + 2 = 3 := by sorry

<informal_problem>
{informal_problem}
</informal_problem>
"""

prompt_translate_sketch_only = """
Given an informal proof written in natural language, translate it into a formal proof in Lean4.

The problem written in the informal natural language is in the "<informal_problem>" tag.
The problem written in the formal Lean4 language is in the "<formal_problem>" tag.
The proof written in the informal natural language is in the "<informal_proof>" tag.

Your output should be a Lean4 code which is the formal proof that can pass Lean4 compilation.
Directly output the code without anything else.
The formal proof in your output should match the provided informal proof, step by step.

Expected format for each step:
1. First, add a comment explaining what the step is doing.
2. Define a `have` statement to describe what is being achieved in this step.
3. For now, no need to write detailed reasoning step. Simply use `by sorry` for each have.

Before the "have-by" steps in the formal proof, there should not be any extra step, such as "intro".
You should avoid using the "intro" tactic.

<informal_problem>
{informal_problem}
</informal_problem>

<formal_problem>
{formal_problem}
</formal_problem>

<informal_proof>
{informal_proof}
</informal_proof>
"""

prompt_translate_sketch_and_partial_proof = """
Given an informal proof written in natural language, translate it into a formal proof in Lean4.

The problem written in the informal natural language is in the "<informal_problem>" tag.
The problem written in the formal Lean4 language is in the "<formal_problem>" tag.
The proof written in the informal natural language is in the "<informal_proof>" tag.

Your output should be a Lean4 code which is the formal proof that can pass Lean4 compilation.
Directly output the code without anything else.
The formal proof in your output should match the provided informal proof, step by step.

Expected format for each step:
1. First, add a comment explaining what the step is doing.
2. Define a `have` statement to describe what is being achieved in this step.
3. Under the `have` statement, try to include the proof if and only if it exists in the given informal proof. In case that the corresponding proof for the step is missing or incomplete in the given informal proof, you may leverage the "sorry" tactic to indicate that. Overall, you should try to make the step in the formal proof a faithful translation of the corresponding step in the informal proof, without any addition or deletion.

<informal_problem>
{informal_problem}
</informal_problem>

<formal_problem>
{formal_problem}
</formal_problem>

<informal_proof>
{informal_proof}
</informal_proof>
"""

prompt_translate = {
    "v1": prompt_translate_sketch_only,
    "v2": prompt_translate_sketch_and_partial_proof,
}

prompt_header = r"""
Start your output with the header shown in the "<header>" tag:
<header>
{header}
</header>
"""

header_default = r"""
import Mathlib
import Aesop

open BigOperators Real Nat Topology Rat
"""

# A simple version for training.
# A longer one is at https://github.com/ucr-rai/formal_verify_step-by-step/blob/14ac0cd2f04b0415570c5c7d0095f75ee7439002/math500lean/rf_prompt_answer_replacement_v2.txt
prompt_fill_answer_json = r"""
You are given an informal problem statement (in the "<informal_problem>" tag) in natural language.
You are also given a formal problem statement (in the "<formal_problem>" tag) in Lean 4, which aims to prove that a certain answer (in the "<original_answer>" tag) is correct for the original problem.

Your task has two parts that you must answer together as one JSON object:

1. Identify the answer-dependent local block in the formal statement: a contiguous substring of <formal_problem> whose content depends on the original answer. It must appear verbatim and exactly once in the formal statement.

2. Write a Python function `def fill_answer(answer: str) -> str:` that, given any new answer string with the same shape as <original_answer>, returns the replacement substring for the answer-dependent block. The function must return ONLY the replacement substring, NOT the entire formal statement. The caller will perform `formal_problem.replace(original_local_block, fill_answer(answer), 1)` to obtain the new formal statement.

Return only a JSON object with these fields:
- "original_local_block": the exact substring of formal_problem that depends on the original answer.
- "fill_answer_code": a Python source string of a `def fill_answer(answer: str) -> str:` function. The function must satisfy `fill_answer(<original_answer>) == <original_local_block>` exactly.

Requirements:
- "original_local_block" must be a contiguous substring of <formal_problem> appearing exactly once.
- The function must handle variations in answer formatting (whitespace, brace styles, separators) robustly.
- Do not output the full formal statement.
- Do not output prose, markdown, or anything outside the JSON object.

<informal_problem>
{informal_problem}
</informal_problem>

<formal_problem>
{formal_problem}
</formal_problem>

<original_answer>
{original_answer}
</original_answer>
"""

prompt_replacement_function = r"""
You are given an informal problem statement (in the "<informal_problem>" tag) in natural language.
You are also given a formal problem statement (in the "<formal_problem>" tag) in Lean 4, which aims to prove that a certain answer (in the "<answer>" tag) is correct for the original problem.
Your task is to write a Python function that takes any new answer and returns a new formal problem statement, with the answer in the problem statement replaced by the new answer, while keeping the rest of the problem statement unchanged.
The function should start with `def fill_answer(answer: str) -> str:`.

<informal_problem>
{informal_problem}
</informal_problem>

<formal_problem>
{formal_problem}
</formal_problem>

<answer>
{answer}
</answer>
"""

prompt_replacement_location_function = r"""
You are given an informal problem statement (in the "<informal_problem>" tag) in natural language.
You are also given a formal problem statement (in the "<formal_problem>" tag) in Lean 4, which aims to prove that a certain answer is correct for the original problem.
You are also given the answer reflected in the formal statement (in the "<answer>" tag).

Your task is to locate the answer span inside the formal problem and write a Python function that only converts a new answer into the formal answer fragment for that span.
The final formal problem will be reconstructed by the caller as:

    prefix + fill_answer(answer) + suffix

where prefix and suffix are copied exactly from the original formal problem using the answer span location.

Return only a JSON object with these fields:
- "answer_start": the 0-based character start offset of the answer span in the formal problem.
- "answer_end": the 0-based character end offset of the answer span in the formal problem.
- "answer_span": the exact answer span copied from the formal problem.
- "python_function": a Python function starting with `def fill_answer(answer: str) -> str:` that returns only the replacement answer fragment, not the full formal problem.

<informal_problem>
{informal_problem}
</informal_problem>

<formal_problem>
{formal_problem}
</formal_problem>

<answer>
{answer}
</answer>
"""

prompt_replacement_context_function = r"""
You are given an informal problem statement (in the "<informal_problem>" tag) in natural language.
You are also given a formal problem statement (in the "<formal_problem>" tag) in Lean 4, which aims to prove that a certain answer is correct for the original problem.
You are also given the answer reflected in the formal statement (in the "<answer>" tag).

Your task is to identify the answer span inside the formal problem without using character offsets.
Instead, return a short exact prefix immediately before the answer span, the exact answer span, and a short exact suffix immediately after the answer span.

The caller will find this exact text in the formal problem:

    prefix_hint + answer_span + suffix_hint

Then it will reconstruct a new formal problem as:

    formal_problem.replace(prefix_hint + answer_span + suffix_hint,
                           prefix_hint + fill_answer(answer) + suffix_hint,
                           1)

Return only a JSON object with these fields:
- "prefix_hint": exact text copied from the formal problem immediately before the answer span.
- "answer_span": the exact answer span copied from the formal problem.
- "suffix_hint": exact text copied from the formal problem immediately after the answer span.
- "python_function": a Python function starting with `def fill_answer(answer: str) -> str:` that returns only the replacement answer fragment, not the full formal problem.

Keep prefix_hint and suffix_hint short, but specific enough that prefix_hint + answer_span + suffix_hint appears exactly once in the formal problem.

<informal_problem>
{informal_problem}
</informal_problem>

<formal_problem>
{formal_problem}
</formal_problem>

<answer>
{answer}
</answer>
"""

prompt_fill_answer_synthesis = r"""
You are given an informal math problem, its Lean 4 formal statement for the original answer, the original candidate answer, and a set of worked examples showing how specific new answers map to local block rewrites of the formal statement.

Your task is to write a Python function `fill_answer(answer: str) -> str` that:
- Parses the input answer string and extracts its structured components.
- Returns only the Lean-side local block that encodes this answer, matching the style of the worked examples.
- Satisfies every worked example exactly (exact string equality).
- Generalizes to unseen answers with the same structural shape as the examples.

Important:
- Do not output the full formal statement.
- Return only the local block substring that replaces the answer-dependent region of the formal statement.
- The function must handle variations in answer formatting (whitespace, braces, separators) robustly.

<informal_problem>
{informal_problem}
</informal_problem>

<formal_problem>
{formal_problem}
</formal_problem>

<original_answer>
{original_answer}
</original_answer>

<worked_examples>
{worked_examples}
</worked_examples>

Return only the Python function, starting with `def fill_answer(answer: str) -> str:`. No markdown fences, no prose.
"""

prompt_block_rewrite = r"""
You are given an informal math problem, a Lean 4 formal statement produced for one candidate answer, the original candidate answer, and a new candidate answer.

The original answer does not appear literally in the formal statement. Instead, the answer controls a local block of the Lean statement. This block may encode a composite or structured answer through a conjunction, list, logical polarity, or symbolic choice.

Your task:
1. Read the new answer and reconstruct its Lean-side structure.
2. Find the answer-dependent local block in the original formal statement.
3. Rewrite that block so it encodes the new answer, preserving the surrounding statement exactly.

Return only a JSON object with these fields:
- "normalized_answer_structure": a concise Lean-style representation of the new answer (for example, a list of pairs, a conjunction form, or an explicit polarity).
- "original_local_block": the exact substring of the formal statement that depends on the original answer. It must appear verbatim in the formal statement.
- "rewritten_local_block": the replacement substring that encodes the new answer, following the same Lean style as the original.

Requirements:
- Do not output the full formal statement.
- Do not output Python code.
- "original_local_block" must be a contiguous substring of the formal statement.
- The caller will substitute the local block via formal_problem.replace(original_local_block, rewritten_local_block, 1), so the original block should appear exactly once in the formal statement.
- Keep both blocks as short as possible while capturing all answer-dependent tokens.

<informal_problem>
{informal_problem}
</informal_problem>

<formal_problem>
{formal_problem}
</formal_problem>

<original_answer>
{original_answer}
</original_answer>

<new_answer>
{new_answer}
</new_answer>
"""

prompt_answer_translation_expression = r"""
You are given an informal math problem, a Lean 4 formal statement produced for one candidate answer, the original candidate answer, and a new candidate answer.

In some cases, the original answer is not preserved literally in the Lean statement because it was converted into a different Lean-style mathematical expression.
Your task is to recover the Lean-side answer expression and translate the new answer into the corresponding Lean-style expression.

Return only a JSON object with these fields:
- "original_lean_answer_expr": the Lean-side expression inside the formal statement that corresponds to the original answer.
- "new_lean_answer_expr": the corresponding Lean-side expression for the new answer, written in a style consistent with the formal statement.
- "reason": one short sentence explaining the conversion pattern.

Requirements:
- Do not output the full formal statement.
- Do not output Python code.
- Keep the expressions as short as possible.
- Reuse the notation and style already present in the formal statement.
- If the formal statement already uses a normalized Lean expression, prefer that style over the raw informal answer text.

<informal_problem>
{informal_problem}
</informal_problem>

<formal_problem>
{formal_problem}
</formal_problem>

<original_answer>
{original_answer}
</original_answer>

<new_answer>
{new_answer}
</new_answer>
"""

prompt_replacement_candidate_function = r"""
You are given an informal problem statement (in the "<informal_problem>" tag) in natural language.
You are also given a formal problem statement (in the "<formal_problem>" tag) in Lean 4, which aims to prove that a certain answer is correct for the original problem.
You are also given the answer reflected in the formal statement (in the "<answer>" tag).

The caller has already found every exact occurrence of the answer string inside the formal problem. These occurrences are listed in "<candidate_answer_occurrences>".

Your task:
1. Decide which candidate occurrence(s) are actual answer slots that should be replaced when a new answer is supplied.
2. Write a Python function that converts a new answer into the formal answer fragment for those selected slots.

Important:
- Do not select occurrences that are merely constants from the problem statement or unrelated assumptions.
- Select multiple ids only if multiple occurrences should be synchronized to the new answer.
- If exactly one candidate is the answer slot, select only that one.
- Do not output character offsets or copy the full Lean statement.
- The Python function must return only the replacement answer fragment, not the full formal problem.

Return only a JSON object with these fields:
- "selected_target_ids": a list of integer candidate ids to replace.
- "python_function": a Python function starting with `def fill_answer(answer: str) -> str:`.

Example output:
{{
  "selected_target_ids": [1],
  "python_function": "def fill_answer(answer: str) -> str:\n    return answer.strip()"
}}

<informal_problem>
{informal_problem}
</informal_problem>

<formal_problem>
{formal_problem}
</formal_problem>

<answer>
{answer}
</answer>

<candidate_answer_occurrences>
{candidates_json}
</candidate_answer_occurrences>
"""
