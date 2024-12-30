#practical 05-
import numpy as np

def is_linearly_dependent(vectors):
    # Convert the list of vectors to a numpy matrix (each vector is a column)
    A = np.column_stack(vectors)

    # Perform row reduction (Gaussian elimination)
    rank = np.linalg.matrix_rank(A)

    # If the rank of the matrix is less than the number of vectors, they are linearly dependent
    if rank < A.shape[1]:
        print("The vectors are linearly dependent.")
        # Find a non-trivial solution to the equation A * c = 0
        # where c is the vector of coefficients
        # Solve A * c = 0 (using least squares)
        c = np.linalg.lstsq(A, np.zeros(A.shape[0]), rcond=None)[0]
        print("Non-trivial linear combination (coefficients):")
        print(c)
    else:
        print("The vectors are linearly independent.")

def generate_linear_combination(vectors, coefficients):
    # Linear combination of vectors based on given coefficients
    result = np.zeros_like(vectors[0])
    for i, vec in enumerate(vectors):
        result += coefficients[i] * vec
    return result

# Example: 3 vectors in R^3 (3D space)
v1 = np.array([1, 2, 3])
v2 = np.array([2, 4, 6])
v3 = np.array([3, 6, 9])

vectors = [v1, v2, v3]

# Check if the vectors are linearly dependent
is_linearly_dependent(vectors)

# Example: Generating a linear combination
coefficients = np.array([2, -1, 1])  # coefficients for the linear combination
linear_combination = generate_linear_combination(vectors, coefficients)
print("\nGenerated linear combination:")
print(linear_combination)


practical -06
import numpy as np
from scipy.linalg import eig
from sympy import Matrix, symbols, det

# Function to check diagonalizability
def is_diagonalizable(A):
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = eig(A)
    
    # Check the geometric multiplicity (number of independent eigenvectors)
    # If the number of independent eigenvectors matches the size of the matrix, it's diagonalizable
    rank = np.linalg.matrix_rank(eigenvectors)
    if rank == A.shape[0]:
        return True, eigenvalues
    else:
        return False, eigenvalues

# Function to verify the Cayley-Hamilton theorem
def verify_cayley_hamilton(A):
    # Convert matrix to sympy Matrix
    A_sympy = Matrix(A)
    
    # Compute the characteristic polynomial
    lambda_symbol = symbols('lambda')
    char_poly = A_sympy.charpoly(lambda_symbol)
    
    # The characteristic polynomial as a sympy expression
    characteristic_polynomial = char_poly.as_expr()

    # Replace lambda with the matrix A in the characteristic polynomial
    A_substitution = characteristic_polynomial.subs(lambda_symbol, A_sympy)
    
    # Check if the result is the zero matrix (Cayley-Hamilton should hold)
    return A_substitution.is_zero

# Example matrix
A = np.array([[4, -1, 1],
              [-1, 4, -2],
              [1, -2, 3]])

# Step 1: Check if the matrix is diagonalizable
diagonalizable, eigenvalues = is_diagonalizable(A)
print("Is the matrix diagonalizable?", diagonalizable)
print("Eigenvalues of the matrix:", eigenvalues)

# Step 2: Verify Cayley-Hamilton theorem
is_cayley_hamilton_true = verify_cayley_hamilton(A)
print("Does the matrix satisfy the Cayley-Hamilton theorem?", is_cayley_hamilton_true)

practical -07
import sympy as sp

# Define symbols (coordinates)
x, y, z = sp.symbols('x y z')

# Example Scalar Field f(x, y, z)
f = x**2 + y**2 + z**2

# Example Vector Field A(x, y, z)
A_x = x * y
A_y = y * z
A_z = z * x
A = sp.Matrix([A_x, A_y, A_z])

# 1. Compute the Gradient of the scalar field f
gradient_f = sp.Matrix([sp.diff(f, var) for var in (x, y, z)])
print("Gradient of f(x, y, z):")
sp.pprint(gradient_f)

# 2. Compute the Divergence of the vector field A
divergence_A = sp.diff(A_x, x) + sp.diff(A_y, y) + sp.diff(A_z, z)
print("\nDivergence of A(x, y, z):")
sp.pprint(divergence_A)

# 3. Compute the Curl of the vector field A
curl_A = sp.Matrix([
    sp.diff(A_z, y) - sp.diff(A_y, z),  # i-component
    sp.diff(A_x, z) - sp.diff(A_z, x),  # j-component
    sp.diff(A_y, x) - sp.diff(A_x, y)   # k-component
])
print("\nCurl of A(x, y, z):")
sp.pprint(curl_A)

