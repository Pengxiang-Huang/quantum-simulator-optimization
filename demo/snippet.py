import time
import numpy as np
from numpy import kron, matmul, array, eye, absolute, real, imag, log2, zeros, sqrt
SWAP = array([
  [1, 0, 0, 0],
  [0, 0, 1, 0],
  [0, 1, 0, 0],
  [0, 0, 0, 1]
])
CNOT = array([
  [1, 0, 0, 0], [0, 1, 0, 0],
  [0, 0, 0, 1],
  [0, 0, 1, 0]
])
isq2 = 1.0 / (2.0 ** 0.5)
H = isq2 * array([
  [1,  1],
  [1, -1],
])
def Id(i): return eye(i, dtype=complex)
def Init(i):
  s = zeros(2 ** i, dtype=complex)
  s[0] = 1
  return s
def print_binary(index, size_sqrt):
  bin_format = f"0{int(log2(size_sqrt**2))}b"
  binary = format(index, bin_format)
  print(f"{binary}", end="")
def print_result(arr, size):
  print("[", end="")
  for i in range(size):
    if absolute(real(arr[i])) < 1e-18 and imag(arr[i]) == 0.0:
      continue
    print(arr[i], end="")
    print("|", end="")
    print_binary(i, sqrt(size))
    print("⟩", end="")
    if i < size - 1:
      print(", ", end="")
  print("]")

#############
start_time = time.time()
x1 = Id(1)
x2 = Id(8)
x3 = kron(kron(x1, H), x2)
x4 = Init(4)
x4 = matmul(x3, x4)
x5 = Id(2)
x6 = Id(4)

x4 = matmul(kron(kron(x5, H), x6), x4)
x4 = matmul(kron(kron(x1, SWAP), x6), x4)
x4 = matmul(kron(kron(x5, CNOT), x5), x4)
x4 = matmul(kron(kron(x6, SWAP), x1), x4)
x4 = matmul(kron(kron(x5, CNOT), x5), x4)
x4 = matmul(kron(kron(x1, SWAP), x6), x4)
x4 = matmul(kron(kron(x5, SWAP), x5), x4)
x4 = matmul(kron(kron(x6, CNOT), x1), x4)
x4 = matmul(kron(kron(x5, SWAP), x5), x4)
x4 = matmul(kron(kron(x5, CNOT), x5), x4)
x4 = matmul(kron(kron(x1, H), x2), x4)
x4 = matmul(kron(kron(x5, H), x6), x4)
()
print("--- %s seconds ---" % (time.time() - start_time))

print_result(x4, 2**4)

