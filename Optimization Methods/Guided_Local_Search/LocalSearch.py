import time
from typing import List, Tuple, Dict
from collections.abc import Callable
import matplotlib.pyplot as plt
import numpy as np


# TO-DO:
# 1. Нужно переписать условия выхода из цикла +
# 2. Нужно понять как определять минимумы аргументов +
# 3. Детальнее вникнуть в статью
# 4. Написать GLS


class LocalSearch:
  def __init__(self, nsteps: int = 1000, eps: float = 1e-6, step: float = .1) -> None:
    self.nsteps = nsteps
    self.eps = eps
    self.step = step
    self.history = []

  def _root(self, x: np.array, f: Callable[[np.array], np.array]) -> np.array:
    self.status = True
    n = 0
    size = x.shape
    ksi = self._get_ksi(size)
    while True:
      f_i = f(x)
      x_new = x.copy()
      x_prev = x.copy()
      for i in range(size[-1]):
        if i >= 1:
          x[i-1] = x_prev[i-1]
        x[i] += self.step * ksi[i]
        f_ii = f(x)
        if f_ii - f_i >= 0:
          ksi[i] = self._get_ksi(size)[i]
        else:
          x_new[i] = x[i]
      x = x_new
      if n >= self.nsteps:
        self.status = False
        break
      if self._check_local_min(x, f):
        break
      n += 1
    return x, self.status, n

  def _check_local_min(self, x: np.array, f: Callable[[np.array], np.array]) -> bool:
    status = True
    x_p = x.copy()
    x_m = x.copy()
    for i in range(len(x)):
      x_p[i] = x_p[i] + 100*self.eps
      x_m[i] = x_m[i] - 100*self.eps
      if (f(x) > f(x_m)) | (f(x) > f(x_p)):
        status = False
    return status

  def _get_ksi(self, size: int) -> np.array:
    ksi = np.ones(size)
    for i in range(size[-1]):
      dist = np.random.uniform(-1,1,1)
      if dist <= 0.:
        ksi[i] *= -1.
    return ksi


def f(x: np.array) -> float:
  return np.sum(np.power(x, 2), -1)


if __name__ == '__main__':
  ls = LocalSearch()
  x = np.array([-20, 12])
  root, status, n = ls._root(x, f)
  print(root, status, n, '\n')