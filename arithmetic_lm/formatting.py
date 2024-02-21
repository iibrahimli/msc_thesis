"""
Input formatting types for arithmetic tasks:
 - Plain: standard formatting of addition
 - Reverse: flips the order of the output and encapsulates each data sample with the '$' symbol at the start and end.
 - Simplified Scratchpad: provides carry and digit-sum information for each step of addition, from the LSB to the MSB.
 - Detailed Scratchpad: provides explicit details of intermediate steps of addition.
"""

PLAIN_FORMAT_STR = "{a}{op}{b}={ans}\n"
PAD_FORMAT_STR = "${a}{op}{b}={ans}$\n"

# TODO
