"""
Input formatting types for arithmetic tasks
"""

import random
import re
import string

from arithmetic_lm.constants import TASK_PREFIX_LEN

PLAIN_FORMAT_STR = "{a}{op}{b}={ans}\n"

INDEX_HINTS = string.ascii_lowercase


def add_random_spaces(text: str, amount: int | float = 0.1) -> str:
    """
    Add random spaces to a text in random positions.

    Args:
        text (str): The text to add spaces to.
        amount (int | float): The amount of spaces to add. If int, it will add
        that number of spaces. If float, it will add that ratio of spaces wrt
        text length.
    """

    max_n_spaces = amount if isinstance(amount, int) else int(len(text) * amount)
    n_spaces = random.randint(0, max_n_spaces)

    for _ in range(n_spaces):
        pos = random.randint(0, len(text))
        text = text[:pos] + " " + text[pos:]

    return text


def split_operands_and_op(prompt: str) -> tuple[str, str, str]:
    prompt = prompt.replace(" ", "")
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
    scratchpad: bool = False,
    operand_random_spaces_amount: int | float = 0,
    answer_random_spaces_amount: int | float = 0,
    index_hints: bool = False,
    generic: bool = False,
    use_task_prefix: bool = False,
) -> str:
    """
    Format line based on args, assumes line ends with \n. pad_ans_zero is the number of digits
    to pad with zeros to, e.g. 43 -> 043 if pad_ans_zero=3. pad_ops_zero is
    the number of digits to pad the operands with zeros to, e.g. 43+3 -> 043+003
    if pad_ops_zero=3. If random_zero_padding is True, pad operands and answers
    with a random number of zeros between length of number and pad_*_zero.
    filler_tokens_* is the number of filler tokens to prepend before the prompt/ans.
    generic: whether to only apply pad, do not try to split numeric ops and answer.
    task_prefix: (for multi-task) if there are task prefixes in beginning of lines
    NOTE: task prefix assumed to be first 3 chars in each line.
    index hints: "a1b2c3+a4b5c6"
    """

    task_prefix = ""

    if use_task_prefix:
        task_prefix = line[:TASK_PREFIX_LEN]
        line = line[TASK_PREFIX_LEN:]

    # HACK if non-numeric (e.g. matching)
    if generic:
        return f"{task_prefix}{pad}{line.strip()}{pad}"

    ab, ans = line.split("=")
    ans = ans.rstrip()

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

    if reverse_ans:
        ans = ans[::-1]
    if pad_ans_zero:
        ans = ans.zfill(pad_ans_zero)

    # add index hints to ops and ans
    if index_hints:
        # choose a random subsequence of index hints, but one that's enough for
        n_hints = max(len(a), len(b), len(ans))
        start_idx = random.randint(0, len(INDEX_HINTS) - n_hints)

        # ans may be longer than ops by 1, so adjust start_idx
        a = "".join(
            [
                f"{ih}{da}"
                for da, ih in zip(a, INDEX_HINTS[start_idx + int(not reverse_ans) :])
            ]
        )
        b = "".join(
            [
                f"{ih}{da}"
                for da, ih in zip(b, INDEX_HINTS[start_idx + int(not reverse_ans) :])
            ]
        )
        ans = "".join([f"{ih}{da}" for da, ih in zip(ans, INDEX_HINTS[start_idx:])])

    ab = f"{a}{op}{b}"
    ab = ab.lstrip()

    # add random spaces to operands
    ab = add_random_spaces(ab, operand_random_spaces_amount)

    # add random spaces to answer
    ans = add_random_spaces(ans, answer_random_spaces_amount)

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

    if scratchpad:
        scratchpad = scratchpad_addition(a, b)
        res = f"{pad}{ab}={scratchpad}{pad}"
    else:
        res = f"{pad}{ab}={ans}{pad}"

    if prepend_newline:
        res = "\n" + res
    if append_newline:
        res += "\n"

    res = task_prefix + res

    return res


def scratchpad_addition(a: str, b: str) -> str:
    """
    Generate scratchpad of digit-wise addition and keeping track of carries
    Input: 567+7890
    scratchpad: 7 6 5 0 + 0 9 8 7;c=0,7+0+0=7,c=0;6+9+0=5,c=1;5+8+1=4,c=1;0+7+1=8,c=0|8457
    """

    res = ""

    length = max(len(a), len(b))
    a = a.zfill(length)
    b = b.zfill(length)

    # add reversed numbers
    res += " ".join(a[::-1]) + " + " + " ".join(b[::-1]) + ";c=0,"

    carry = 0

    # start from last digit
    for da, db in zip(reversed(a), reversed(b)):
        da = int(da)
        db = int(db)
        msum = (da + db + carry) % 10
        res += f"{da}+{db}+{carry}={msum},"
        carry = (da + db + carry) // 10
        res += f"c={carry};"
    res = res[:-1]  # remove last comma
    res += f"|{int(a)+int(b)}"
    return res
