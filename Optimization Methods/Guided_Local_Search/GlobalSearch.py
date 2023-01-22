import numpy as np
from typing import List, Tuple, Dict
from collections.abc import Callable
from LocalSearch import LocalSearch


class GlobalSearch(LocalSearch):
  def __init__(self, nsteps: int = 10000, eps: float = 1e-5, step: float = 1e-1) -> None:
    super().__init__(nsteps, eps, step)
    self.l = 0.4
    self.c_criteria = 2.4

  def root(self, x: np.array, f: Callable[[np.array], np.array]) -> np.array:
    size = x.shape
    self.p = np.zeros(size)
    self.util = np.zeros(size)
    self.c = self.c_criteria * np.ones(size)
    self.I = np.zeros(size)
    self.s = self.s_by_f = [f(x)]
    k = 0
    h = self._cost_function(self.s[-1])
    delta_f = 2 * self.eps
    while (k <= self.nsteps) & (delta_f >= self.eps):
      s_i, status = self._root(self.s[-1], h)
      self.s_by_f.append(f(s_i))
      self.s.append(s_i)
      for i in range(len(size)):
        self.util[i] = status[i] * self.c[i] / (1 + self.p[i])
      indexes = np.argmax(self.util, -1)
      self.p[indexes] += 1
      k += 1
    return self.s[np.argmin(self.s_by_f)[0]]

  def _cost_function(self, cost: float) -> np.array:
    return cost + self.l * (self.p @ self.I)


def f(x: np.array) -> float:
  return np.sum(np.power(np.cos(x) - 2 * np.ones(x.shape), 2))


if __name__ == "__main__":
  gls = GlobalSearch()
  x = np.array([0., -10.])
  root = gls.root(x, f)