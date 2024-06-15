"""
Input formatting types for arithmetic tasks
"""

import random
import re

PLAIN_FORMAT_STR = "{a}{op}{b}={ans}\n"


def split_operands_and_op(prompt: str) -> tuple[str, str, str]:
    if "=" in prompt:
        prompt = prompt[: prompt.index("=")]
    a, b = re.findall(r"\d+", prompt)
    op = re.findall(r"[+\-*]", prompt)[0]
    return a, op, b


def format_line(
    line: str,
    pad: str = None,
    reverse_ops: bool = False,
    reverse_ans: bool = False,
    pad_ops_zero: int | None = None,
    pad_ans_zero: int | None = None,
    filler_tokens_prompt: int | None = None,
    filler_tokens_ans: int | None = None,
    random_fillers: bool = False,
    prepend_newline: bool = False,
    append_newline: bool = False,
    random_zero_padding: bool = False,
    chain_of_thought: bool = False,
    generic: bool = False,
) -> str:
    """
    Format line based on args, assumes line has ends with \n,
    returned line ends with \n as well. pad_ans_zero is the number of digits
    to pad with zeros to, e.g. 43 -> 043 if pad_ans_zero=3. pad_ops_zero is
    the number of digits to pad the operands with zeros to, e.g. 43+3 -> 043+003
    if pad_ops_zero=3. If random_zero_padding is True, pad operands and answers
    with a random number of zeros between length of number and pad_*_zero.
    filler_tokens_* is the number of filler tokens to prepend before the prompt/ans.
    generic: whether to only apply pad, do not try to split numeric ops and answer.
    """

    # HACK if non-numeric (e.g. matching)
    if generic:
        return f"{pad}{line}{pad}"

    ab, ans = line.split("=")

    if random_zero_padding:
        assert (
            pad_ops_zero is not None and pad_ans_zero is not None
        ), "pad_ops_zero and pad_ans_zero must be provided if random_zero_padding is True"
        pad_ops_zero = random.randint(0, pad_ops_zero)
        pad_ans_zero = random.randint(0, pad_ans_zero)

    a, op, b = split_operands_and_op(ab)

    if reverse_ops:
        a = a[::-1]
        b = b[::-1]

    if pad_ops_zero:
        # split by non-digit char and pad operands with zeros
        a = a.zfill(pad_ops_zero)
        b = b.zfill(pad_ops_zero)

    ab = f"{a}{op}{b}"
    ab = ab.lstrip()
    ans = ans.rstrip()

    if reverse_ans:
        ans = ans[::-1]

    if pad_ans_zero:
        ans = ans.zfill(pad_ans_zero)

    filler_token = "."
    if filler_tokens_prompt:
        if random_fillers:
            filler_tokens_prompt = random.randint(0, filler_tokens_prompt)
        ab = filler_token * filler_tokens_prompt + ab
    if filler_tokens_ans:
        if random_fillers:
            filler_tokens_ans = random.randint(0, filler_tokens_ans)
        ans = filler_token * filler_tokens_ans + ans

    pad = pad if pad else ""

    if chain_of_thought:
        cot = chain_of_thought_addition(a, b)
        res = f"{pad}{ab}={cot}{pad}"
    else:
        res = f"{pad}{ab}={ans}{pad}"

    if prepend_newline:
        res = "\n" + res
    if append_newline:
        res += "\n"

    return res


def chain_of_thought_addition(a: str, b: str) -> str:
    """
    Input: 567+7890
    CoT: 7+0=7c0,6+9=5c1,5+8=3c1,0+7=8c0|567+7890=8457
    """
    res = ""

    length = max(len(a), len(b))
    a = a.zfill(length)
    b = b.zfill(length)

    # start from last digit
    for da, db in zip(reversed(a), reversed(b)):
        da = int(da)
        db = int(db)
        msum = (da + db) % 10
        carry = (da + db) // 10
        res += f"{da}+{db}={msum}c{carry},"
    res = res[:-1]  # remove last comma
    res += f"|{a}+{b}={int(a)+int(b)}"
    return res
