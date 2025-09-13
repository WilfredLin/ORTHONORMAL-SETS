import numpy as np

def is_orthonormal(vectors, tol=1e-10):
    """Check if a set of vectors is orthonormal (handles float/int inputs)"""
    n = len(vectors)
    for i in range(n):
        # Convert to float for safe norm calculation
        vi = np.array(vectors[i], dtype=float)
        if abs(np.linalg.norm(vi) - 1.0) > tol:
            return False
        
        for j in range(i+1, n):
            vj = np.array(vectors[j], dtype=float)
            if abs(np.dot(vi, vj)) > tol:
                return False
    return True

def gram_schmidt(vectors, tol=1e-10):
    """
    Fixed Gram-Schmidt: Ensures all operations use float dtype
    Converts input vectors to float at the start
    """
    ortho_set = []
    
    # Convert ALL input vectors to float FIRST
    vectors = [np.array(v, dtype=float) for v in vectors]
    
    for v in vectors:
        proj = np.zeros_like(v)  # Now guaranteed float
        for u in ortho_set:
            proj += np.dot(v, u) * u  # Safe float addition
        
        w = v - proj
        norm_w = np.linalg.norm(w)
        
        if norm_w < tol:
            raise ValueError("Input vectors are linearly dependent!")
        
        u = w / norm_w
        ortho_set.append(u)
    
    return ortho_set

# Example Usage (FIXED TEST CASES)
if __name__ == "__main__":
    # Test Case 1: Standard basis (explicit float conversion)
    standard_basis = [
        np.array([1.0, 0, 0]),
        np.array([0, 1.0, 0]),
        np.array([0, 0, 1.0])
    ]
    
    print("Test Case 1: Standard Basis")
    print("Is orthonormal?", is_orthonormal(standard_basis))  # True
    print("-" * 50)
    
    # Test Case 2: Custom orthonormal set (floats)
    custom_set = [
        np.array([1/np.sqrt(2), 1/np.sqrt(2), 0.0]),
        np.array([-1/np.sqrt(2), 1/np.sqrt(2), 0.0]),
        np.array([0.0, 0.0, 1.0])
    ]
    
    print("Test Case 2: Custom Orthonormal Set")
    print("Is orthonormal?", is_orthonormal(custom_set))  # True
    print("-" * 50)
    
    # Test Case 3: Non-orthonormal set (now works with integers!)
    non_ortho_set = [
        np.array([1, 1, 0]),  # Integers are now safely converted
        np.array([1, 0, 1]),
        np.array([0, 1, 1])
    ]
    
    print("Test Case 3: Non-orthonormal Input")
    print("Original set orthonormal?", is_orthonormal(non_ortho_set))  # False
    
    try:
        orthonormal_set = gram_schmidt(non_ortho_set)
        print("\nGram-Schmidt Result:")
        for i, v in enumerate(orthonormal_set):
            print(f"u{i+1} =", np.round(v, 3))
        
        print("\nConverted set orthonormal?", 
              is_orthonormal(orthonormal_set))  # True
        
        # Verification
        print("\nVerification:")
        print("Norms:", [round(np.linalg.norm(v), 3) for v in orthonormal_set])
        print("Dot products (u1·u2, u1·u3, u2·u3):", 
              [round(np.dot(orthonormal_set[i], orthonormal_set[j]), 3) 
               for i in range(2) for j in range(i+1, 3)])
    except ValueError as e:
        print("Error:", e)
