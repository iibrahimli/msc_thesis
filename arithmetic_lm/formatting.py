"""
Input formatting types for arithmetic tasks
"""

import random
import re

PLAIN_FORMAT_STR = "{a}{op}{b}={ans}\n"


def split_operands_and_op(prompt: str) -> tuple[str, str, str]:
    a, b = re.findall(r"\d+", prompt)
    op = re.findall(r"[+\-*]", prompt)[0]
    return a, op, b


def format_line(
    line: str,
    pad: str = None,
    reverse_ans: bool = False,
    pad_ops_zero: int | None = None,
    pad_ans_zero: int | None = None,
    prepend_newline: bool = False,
    append_newline: bool = False,
    random_zero_padding: bool = True,
) -> str:
    """
    Format line based on args, assumes line has ends with \n,
    returned line ends with \n as well. pad_ans_zero is the number of digits
    to pad with zeros to, e.g. 43 -> 043 if pad_ans_zero=3. pad_ops_zero is
    the number of digits to pad the operands with zeros to, e.g. 43+3 -> 043+003
    if pad_ops_zero=3. If random_zero_padding is True, pad operands and answers
    with a random number of zeros between length of number and pad_*_zero.
    """

    ab, ans = line.split("=")

    if pad_ops_zero:
        # split by non-digit char and pad operands with zeros
        a, op, b = split_operands_and_op(ab)
        a = a.zfill(
            pad_ops_zero
            if not random_zero_padding
            else random.randint(len(a), pad_ops_zero)
        )
        b = b.zfill(
            pad_ops_zero
            if not random_zero_padding
            else random.randint(len(b), pad_ops_zero)
        )
        ab = f"{a}{op}{b}"

    ab = ab.lstrip()
    ans = ans.rstrip()

    if reverse_ans:
        ans = ans[::-1]

    if pad_ans_zero:
        ans = ans.zfill(
            pad_ans_zero
            if not random_zero_padding
            else random.randint(len(ans), pad_ans_zero)
        )

    pad = pad if pad else ""
    res = f"{pad}{ab}={ans}{pad}"
    if prepend_newline:
        res = "\n" + res
    if append_newline:
        res += "\n"

    return res
