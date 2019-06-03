import re
import string

import numpy as np
import tensorflow as tf


def tensor_matrix_mul(t, m):
    return tf.reshape(tf.reshape(t, [-1, t.shape[-1]]) @ m, [-1, t.shape[1], m.shape[-1]])


def vec_mat_mul(vector, matrix):
    return tf.multiply(tf.expand_dims(vector, -1), matrix)


def prefix_sum(arr):
    res = [0]
    for a in arr: res.append(res[-1] + a)
    return res


# Identifier check function
id_regex = "^[a-zA-Z$_][a-zA-Z0-9$_]*$"


def is_word(name):
    return bool(re.match(id_regex, name))  # and name not in ts_keywords


punct_regex = re.compile('[%s]' % re.escape(string.punctuation))
num_regex = "([0-9]x?|[\.][0-9])[0-9\.]*(lLfF)?"


def split_subtokens(token):
    token = token.strip()
    if not is_word(token):
        if token.startswith("'"):
            return ("''",)
        elif token.startswith("`") or "`" in token:
            return ("``",)
        elif token.startswith("/"):
            return ("/",)
        elif token.startswith("\"") or " " in token:
            return ("\"\"",)
        # elif token in ts_keywords: return (token,)
        elif re.match(punct_regex, token):
            return (token,)
        elif re.match(num_regex, token):
            return token
        else:
            return (token,)
    parts = segment(token)
    parts = [p.lower() for p in parts if len(p) > 0 and not re.match(punct_regex,
                                                                     p)]  # Drop any punctuation and lower-case the rest, for input tokens specifically
    if len(parts) == 0:
        parts = [token]
    return tuple(parts)


def segment(token):
    def update_splits(lix, new_lix):
        if new_lix == 0 or lix == new_lix: return lix
        split.append(token[lix:new_lix])
        return new_lix

    lix = 0
    split = []
    mode = ""
    for ix, c in enumerate(token):
        if c.islower():  # For any lower-case,
            if mode == "upper" and lix < ix - 1:  # if we have seen several upper-case letters, assume an acronym was involved and only the last upper-case is part of a new sub-token
                lix = update_splits(lix, ix - 1)
            elif mode not in ["lower", "upper"]:  # else, if we were looking at non-letters, split
                lix = update_splits(lix, ix)
            mode = "lower"
        elif c.isupper():  # For upper-case characters,
            if mode != "upper":  # segment if it follows any non-uppercase
                lix = update_splits(lix, ix)
            mode = "upper"
        elif re.match("\s", c):  # For whitespace,
            if mode != "space": update_splits(lix,
                                              ix)  # Only segment if we didn't just skip whitespace, and only up to the previous character
            lix = ix + 1  # Skip whitespace, which will repeat as often as necessary
            mode = "space"  # Simply reset the mode; we start from scratch after whitespace
        elif re.match("[0-9]", c):  # Split on every numeral, regardless of the previous character
            lix = update_splits(lix, ix)
            mode = "num"
        else:  # Otherwise, assume some form of punctuation and split on every occurrence
            lix = update_splits(lix, ix)
            mode = "punct"
    if lix < len(token): split.append(token[lix:])  # Append any remainder
    return split
