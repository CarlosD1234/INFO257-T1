import math

X_MIN = -3.0
X_MAX = 7.0


def rastrigin(x, y):
    return 20 + (x**2 - 10 * math.cos(2 * math.pi * x)) + (y**2 - 10 * math.cos(2 * math.pi * y))


def clamp(val, lo, hi):
    return max(lo, min(hi, val))
