"""
Input formatting types for arithmetic tasks
"""

import re

PLAIN_FORMAT_STR = "{a}{op}{b}={ans}\n"


def format_line(
    line: str,
    pad: str = None,
    reverse_ans: bool = False,
    pad_ops_zero: int | None = None,
    pad_ans_zero: int | None = None,
    prepend_newline: bool = False,
) -> str:
    """
    Format line based on args, assumes line has ends with \n,
    returned line ends with \n as well. pad_ans_zero is the number of digits
    to pad with zeros to, e.g. 43 -> 043 if pad_ans_zero=3. pad_ops_zero is
    the number of digits to pad the operands with zeros to, e.g. 43+3 -> 043+003
    if pad_ops_zero=3.
    """

    ab, ans = line.split("=")

    if pad_ops_zero:
        # split by non-digit char and pad operands with zeros
        a, b = re.findall(r"\d+", ab)
        op = re.sub(r"\d+", "", ab)
        a = a.zfill(pad_ops_zero)
        b = b.zfill(pad_ops_zero)
        ab = f"{a}{op}{b}"

    ab = ab.lstrip()
    ans = ans.rstrip()

    if pad_ans_zero:
        ans = ans.zfill(pad_ans_zero)

    if reverse_ans:
        ans = ans[::-1]

    pad = pad if pad else ""
    res = f"{pad}{ab}={ans}{pad}\n"
    if prepend_newline:
        res = "\n" + res

    return res
