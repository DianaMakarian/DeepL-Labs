import numpy as np
from src.threshold import verify_algorithms

SETS = {
    "A": {
        "scores": np.array([0.9, 0.7, 0.7, 0.4, 0.2, 0.1]),
        "y":      np.array([0,   1,   0,   1,   0,   0  ]),
    },
    "B": {
        "scores": np.array([0.9, 0.6, 0.3]),
        "y":      np.array([0,   0,   0  ]),
    },
    "C": {
        "scores": np.array([0.8] + [0.5] * 11 + [0.2]),
        "y":      np.array([1  ] + [1  ] + [0] * 10 + [0]),
    },
}


def run_toy_verifications() -> None:
    print("=== Перевірка алгоритмів (Набори A, B, C) ===")
    for name, data in SETS.items():
        verify_algorithms(data["y"], data["scores"], label=name)
    print()