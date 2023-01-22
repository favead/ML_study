import numpy as np
from typing import List, Tuple, Dict
from collections.abc import Callable
from LocalSearch import LocalSearch


class GlobalSearch(LocalSearch):
  def __init__(self, nsteps: int = 10000, eps: float = 1e-5, step: float = 1e-1, m: int = 10, \
    l: float = 0.4, c_criteria: float = 2.4) -> None:
    super().__init__(nsteps, eps, step)
    self.l = l
    self.m = m
    self.c_criteria = c_criteria

  def root(self, x: np.array, f: Callable[[np.array], np.array]) -> np.array:
    self.p = np.zeros((self.m))
    self.util = np.zeros((self.m))
    self.c = self.c_criteria * np.ones((self.m))
    self.I = np.zeros((self.m))
    self.s = [x]
    self.s_by_f = []
    k = 0
    h = self._cost_function
    while True:
      x, status, n = self._root(self.s[-1], h)
      self.s_by_f.append(f(self.s[-1]))
      self.s.append(x)
      self.I[k] = int(status)
      for i in range(self.m):
        self.util[i] = self.I[i] * self.c[i] / (1 + self.p[i])
      indxs = np.argmax(self.util, -1)
      self.p[indxs] += 1
      k += 1
      if k >= self.m:
        break
    min_ind = np.argmin(self.s_by_f)
    return self.s[min_ind]

  def _cost_function(self, x: np.array) -> np.array:
    cost = f(x)
    return cost + self.l * (self.p @ self.I)


def f(x: np.array) -> float:
  return np.sum(np.power(np.cos(x) - 2 * np.ones(x.shape), 2))


if __name__ == "__main__":
  gls = GlobalSearch()
  x = np.array([0., -10.])
  root = gls.root(x, f)
  print(root)