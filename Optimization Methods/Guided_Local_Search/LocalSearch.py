import time
from typing import List, Tuple, Dict
from collections.abc import Callable
import matplotlib.pyplot as plt
import numpy as np


class LocalSearch:
  def __init__(self, nsteps: int = 10000, eps: float = 1e-6, step: float = 0.1) -> None:
    self.nsteps = nsteps
    self.eps = eps
    self.step = step
    self.history = []

  def _root(self, x: np.array, f: Callable[[np.array], np.array]) -> np.array:
    n = 0
    delta_f = 2 * self.eps
    while (np.max(np.abs(delta_f)) >= self.eps) & (self.nsteps > n):
      ksi = self._get_ksi(x.shape[0])
      n += 1
      f_i = f(x)
      x_s = x + ksi
      f_s = f(x_s)
      for i in range(len(f_s)):
        if f_s[i] < f_i[i]:
          x[i] = x_s[i]
      delta_f = f_s - f_i
    return x, n

  def _get_ksi(self, size: int) -> np.array:
    ksi = self.step * np.ones((1, size))
    for i in range(size):
      dist = np.random.uniform(-1,1,1)
      if dist <= 0.:
        ksi[i] *= -1.
    return ksi


def f(x: np.array) -> np.array:
  return np.power(x, 2)


if __name__ == '__main__':
  ls = LocalSearch()
  x = np.array([20.])
  root, n = ls.root(x, f)
  print(root, n, '\n')