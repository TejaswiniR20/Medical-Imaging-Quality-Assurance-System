# threshold.py

"""Thresholding for final prediction decisions."""


def apply_threshold(probabilities, threshold=0.5):
    return [p >= threshold for p in probabilities]
