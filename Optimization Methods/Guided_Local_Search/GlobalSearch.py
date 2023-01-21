import numpy as np
from typing import List, Tuple, Dict
from collections.abc import Callable
from LocalSearch import LocalSearch


class GlobalSearch(LocalSearch):
  def __init__(self, nsteps: int = 10000, eps: float = 1e-5, step: float = 1e-1, m: int = 100) -> None:
    super().__init__(nsteps, eps, step)
    self.l = 0.1
    self.m = m
    self.p = np.zeros((self.m))
    self.util = np.zeros((self.m))
    self.c_criteria = 2.4

  def root(self, x: np.array, f: Callable[[np.array], np.array]) -> np.array:
    size = x.shape
    self.c = self.c_criteria * np.ones(size)
    self.I = np.zeros(size)
    self.s = self.s_by_f = [f(x)]
    k = 0
    h = self._cost_function(self.s[-1])
    delta_f = 2 * self.eps
    while (k <= self.nsteps) & (delta_f >= self.eps):
      self.s.append(self._root(self.s[-1], h))
      for i in range(len(self.m)):
        pass
    return 

  def _cost_function(self, cost: np.array) -> np.array:
    return cost + self.l * (self.p @ self.I)