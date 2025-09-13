def main():
    print("="*50)
    print("ORTHONORMAL SET CALCULATOR")
    print("Using Modified Gram-Schmidt Orthogonalization")
    print("="*50)
    
    # Get input dimensions
    try:
        n = int(input("\nEnter number of vectors: "))
        d = int(input("Enter dimension of vectors: "))
        
        if n <= 0 or d <= 0:
            raise ValueError("Dimensions must be positive integers")
        if n > d:
            print("\nWARNING: More vectors than dimension suggests linear dependence!")
            print("The process will fail if vectors are linearly dependent.")
    except ValueError as e:
        print(f"\nError: {e}. Please enter valid positive integers.")
        return

    # Input vectors
    print("\nEnter vectors (space-separated values for each vector):")
    vectors = []
    for i in range(n):
        while True:
            try:
                data = input(f"Vector {i+1}: ").split()
                if len(data) != d:
                    raise ValueError(f"Expected {d} values, got {len(data)}")
                vec = [float(x) for x in data]
                vectors.append(vec)
                break
            except ValueError as e:
                print(f"  Invalid input: {e}. Try again.")

    # Calculate orthonormal set
    try:
        orthonormal = calculate_orthonormal(vectors)
    except ValueError as e:
        print(f"\nERROR: {e}")
        return

    # Display results
    print("\n" + "="*50)
    print("RESULTING ORTHONORMAL SET:")
    print("="*50)
    
    for i, vec in enumerate(orthonormal):
        # Format with 4 decimal places, handling negative zeros
        formatted = [f"{x:.4f}".replace("-0.0000", "0.0000") for x in vec]
        print(f"e{i+1} = [{', '.join(formatted)}]")
    
    # Verification
    print("\n" + "="*50)
    print("VERIFICATION:")
    print("="*50)
    
    # Check norms
    for i, vec in enumerate(orthonormal):
        norm = sum(x*x for x in vec) ** 0.5
        status = "PASS" if abs(norm - 1.0) < 1e-10 else "FAIL"
        print(f"||e{i+1}|| = {norm:.6f} {'✅' if status == 'PASS' else '❌'}")
    
    # Check orthogonality
    for i in range(len(orthonormal)):
        for j in range(i+1, len(orthonormal)):
            dot = sum(orthonormal[i][k] * orthonormal[j][k] for k in range(d))
            status = "PASS" if abs(dot) < 1e-10 else "FAIL"
            print(f"e{i+1} • e{j+1} = {dot:.6e} {'✅' if status == 'PASS' else '❌'}")

def calculate_orthonormal(vectors):
    """
    Compute orthonormal basis using Modified Gram-Schmidt process
    
    Args:
        vectors: List of input vectors (each is list of floats)
        
    Returns:
        List of orthonormal vectors
        
    Raises:
        ValueError: For linear dependence or dimension issues
    """
    n = len(vectors)
    d = len(vectors[0])
    
    # Validate dimensions
    for i, vec in enumerate(vectors):
        if len(vec) != d:
            raise ValueError(f"Vector {i+1} has {len(vec)} dimensions (expected {d})")
    
    orthonormal = []
    
    for i in range(n):
        u = vectors[i][:]  # Current vector
        
        # Subtract projections onto all previous orthonormal vectors
        for j in range(len(orthonormal)):
            # Compute dot product
            dot_product = sum(u[k] * orthonormal[j][k] for k in range(d))
            
            # Subtract projection (Modified Gram-Schmidt)
            u = [u[k] - dot_product * orthonormal[j][k] for k in range(d)]
        
        # Calculate norm
        norm = sum(x*x for x in u) ** 0.5
        
        # Check for linear dependence
        if norm < 1e-10:
            raise ValueError(f"Linear dependence detected at vector {i+1}")
        
        # Normalize
        e = [x / norm for x in u]
        orthonormal.append(e)
    
    return orthonormal

if __name__ == "__main__":
    main()
