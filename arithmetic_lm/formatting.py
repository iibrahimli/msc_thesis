"""
Input formatting types for arithmetic tasks:
 - Plain: standard formatting of addition
 - Reverse: flips the order of the output and encapsulates each data sample with the '$' symbol at the start and end.
 - Simplified Scratchpad: provides carry and digit-sum information for each step of addition, from the LSB to the MSB.
 - Detailed Scratchpad: provides explicit details of intermediate steps of addition.
"""

PLAIN_FORMAT_STR = "{a}{op}{b}={ans}\n"


def format_line(
    line: str,
    pad: str = None,
    reverse_ans: bool = False,
    prepend_newline: bool = False,
) -> str:
    """
    Format line based on args, assumes line has ends with \n,
    returned line ends with \n as well
    """

    ab, ans = line.split("=")

    ab = ab.lstrip()
    ans = ans.rstrip()

    if reverse_ans:
        ans = ans[::-1]

    pad = pad if pad else ""
    res = f"{pad}{ab}={ans}{pad}\n"
    if prepend_newline:
        res = "\n" + res

    return res
