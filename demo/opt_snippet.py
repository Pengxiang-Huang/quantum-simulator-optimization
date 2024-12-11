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
  [1, 0, 0, 0],
  [0, 1, 0, 0],
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



### define the optimization 
# rule 1: kron(Id1, M) = M, and kron(M, Id(1)) = M 
# rule 2: kron(A, M) := f(M), where f is defined below 
def propagate_zeros_kron( A , M):
    """ where A is known matrix in compile time """
    """ propagate the zeros in A and compute statically """
    rows_A, cols_A = A.shape 
    rows_M, cols_M = M.shape 
    
    # allocate a result 
    result_dtype = np.result_type(A,M)
    result = np.zeros((rows_A * rows_M, cols_A * cols_M), dtype=result_dtype)
    
    for i in range(rows_A):
        for j in range(cols_A):
            start_row = i * rows_M 
            start_col = j * cols_M 
            
            value = A[i,j]
            if value != 0:
                result[start_row:start_row+rows_M, start_col:start_col+cols_M] = value * M

    return result  

### constant folding 
x1 = Id(1)
x2 = Id(8)
x3 = propagate_zeros_kron(H, Id(8))
x4 = Init(4)
x4 = matmul(x3, x4)
x5 = Id(2)
x6 = Id(4)

t0 = propagate_zeros_kron(x5, H)
t1 = propagate_zeros_kron(t0, x6)
t2 = SWAP 
t3 = propagate_zeros_kron(t2, x6)
t9 = propagate_zeros_kron(x5, CNOT)
t10 = propagate_zeros_kron(t9, x5)
t11 = propagate_zeros_kron(x6, SWAP)
t12 = propagate_zeros_kron(x5, CNOT)
t13 = propagate_zeros_kron(t12, x5)
t14 = propagate_zeros_kron(SWAP, x6)
t15 = propagate_zeros_kron(x5, SWAP)
t16 = propagate_zeros_kron(t15, x5)
t17 = propagate_zeros_kron(x6, CNOT)
t18 = propagate_zeros_kron(x5, SWAP)
t19 = propagate_zeros_kron(t18, x5)
t20 = propagate_zeros_kron(x5, CNOT)
t21 = propagate_zeros_kron(t20, x5)
t23 = propagate_zeros_kron(x5, H)
t24 = propagate_zeros_kron(t23, x6)
t22 = propagate_zeros_kron(H, x2)

start_time = time.time()
# x4 = matmul(kron(kron(x5, H), x6), x4)
x4 = matmul(t1, x4)

# x4 = matmul(kron(kron(x1, SWAP), x6), x4)
x4 = matmul(t3, x4)

# x4 = matmul(kron(kron(x5, CNOT), x5), x4)
x4 = matmul(t10, x4)

# x4 = matmul(kron(kron(x6, SWAP), x1), x4)
x4 = matmul(t11, x4)

# x4 = matmul(kron(kron(x5, CNOT), x5), x4)
x4 = matmul(t13, x4)

# x4 = matmul(kron(kron(x1, SWAP), x6), x4)
x4 = matmul(t14, x4)

# x4 = matmul(kron(kron(x5, SWAP), x5), x4)
x4 = matmul(t16, x4)

# x4 = matmul(kron(kron(x6, CNOT), x1), x4)
x4 = matmul(t17, x4)

# x4 = matmul(kron(kron(x5, SWAP), x5), x4)
x4 = matmul(t19, x4)

# x4 = matmul(kron(kron(x5, CNOT), x5), x4)
x4 = matmul(t21, x4)

# x4 = matmul(kron(kron(x1, H), x2), x4)
x4 = matmul(t22, x4)

# x4 = matmul(kron(kron(x5, H), x6), x4)
x4 = matmul(t24, x4)
()
print("--- %s seconds ---" % (time.time() - start_time))

print_result(x4, 2**4)

