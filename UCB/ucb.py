from collections import defaultdict

import numpy as np


class LinUCB:
    """Disjoint LinUCB (Li et al. 2010) with a one-hot context over combinations, as in paper §5.3.

    Arm a = the attribute to change in the guide combination. The context x is the one-hot vector of
    the combination being repaired, so each arm's ridge regression A_a = I + sum x x^T, b_a = sum r x
    is diagonal, and for combination c:
        A_a[c, c] = 1 + n_a(c),  b_a[c] = sum of rewards of arm a on c
        p_a(c) = b_a[c] / A_a[c, c] + alpha * sqrt(1 / A_a[c, c])
    Only the diagonal entries are stored, keyed by the combination's pattern string, so the number
    of combinations doesn't need to be known up front.
    """

    def __init__(self, num_arms: int, alpha: float):
        self.num_arms = num_arms
        self.alpha = alpha
        self.A = defaultdict(lambda: np.ones(self.num_arms))
        self.b = defaultdict(lambda: np.zeros(self.num_arms))

    def scores(self, context: str) -> np.ndarray:
        A, b = self.A[context], self.b[context]
        return b / A + self.alpha * np.sqrt(1 / A)

    def choose_arm(self, context: str, allowed_arms: list[int] | None = None) -> int:
        scores = self.scores(context)
        if allowed_arms:
            masked = np.full(self.num_arms, -np.inf)
            masked[allowed_arms] = scores[allowed_arms]
            scores = masked
        print(f"{context=} {scores=}")
        # break ties at random so untried arms are explored in no fixed order
        return int(np.random.choice(np.flatnonzero(scores == scores.max())))

    def update(self, arm: int, context: str, reward: float):
        self.A[context][arm] += 1
        self.b[context][arm] += reward

    def __getstate__(self):
        return {"num_arms": self.num_arms, "alpha": self.alpha, "A": dict(self.A), "b": dict(self.b)}

    def __setstate__(self, state):
        self.__init__(state["num_arms"], state["alpha"])
        self.A.update(state["A"])
        self.b.update(state["b"])
