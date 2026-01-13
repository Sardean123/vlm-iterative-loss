"""
Verify Convergence Detection - Test with toy examples

Tests the convergence detection algorithm with known cases to verify
understanding and correctness.

Usage: python verify_convergence.py
"""

import numpy as np

def cosine_similarity(vec_a, vec_b):
    """Compute cosine similarity."""
    denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    return float((vec_a @ vec_b) / denom) if denom else 0.0

def detect_convergence(embeddings, window_size=3, threshold=0.95):
    """
    Detect convergence in embedding trajectory.
    
    Returns:
        converged (bool): Whether convergence detected
        iteration (int): Iteration at which convergence flagged (-1 if none)
        attractor_type (str): "fixed_point", "limit_cycle_2", or "none"
    """
    if len(embeddings) < window_size + 1:
        return False, -1, "none"
    
    # Fixed point detection
    recent_sims = []
    for idx in range(len(embeddings) - window_size + 1, len(embeddings)):
        sim = cosine_similarity(embeddings[idx], embeddings[idx - 1])
        recent_sims.append(sim)
    
    if recent_sims and np.mean(recent_sims) > threshold:
        return True, len(embeddings) - window_size, "fixed_point"
    
    # Limit cycle detection (period 2)
    if len(embeddings) >= 6:
        cycle_sims = []
        for idx in range(len(embeddings) - 4, len(embeddings), 2):
            if idx - 2 >= 0:
                cycle_sims.append(cosine_similarity(embeddings[idx], embeddings[idx - 2]))
        if len(cycle_sims) >= 2 and np.mean(cycle_sims) > threshold:
            return True, len(embeddings) - 4, "limit_cycle_2"
    
    return False, -1, "none"

def create_fixed_point_trajectory(dim=768, num_iters=50):
    """Create trajectory that converges to a fixed point."""
    base = np.random.randn(dim)
    base = base / np.linalg.norm(base)
    
    trajectory = [base]
    for i in range(1, num_iters):
        # Add decreasing noise
        noise_scale = 0.5 / (i + 1)
        noise = np.random.randn(dim) * noise_scale
        new_vec = trajectory[-1] + noise
        new_vec = new_vec / np.linalg.norm(new_vec)
        trajectory.append(new_vec)
    
    return trajectory

def create_limit_cycle_trajectory(dim=768, num_iters=50):
    """Create trajectory that oscillates between two states."""
    state_a = np.random.randn(dim)
    state_a = state_a / np.linalg.norm(state_a)
    
    state_b = state_a + np.random.randn(dim) * 0.3
    state_b = state_b / np.linalg.norm(state_b)
    
    trajectory = [state_a]
    for i in range(1, num_iters):
        if i % 2 == 1:
            # Move toward state_b
            target = state_b
        else:
            # Move toward state_a
            target = state_a
        
        # Add decreasing noise as we converge
        noise_scale = 0.3 / (i//2 + 1)
        noise = np.random.randn(dim) * noise_scale
        new_vec = target + noise
        new_vec = new_vec / np.linalg.norm(new_vec)
        trajectory.append(new_vec)
    
    return trajectory

def create_diverging_trajectory(dim=768, num_iters=50):
    """Create trajectory that doesn't converge."""
    trajectory = []
    for i in range(num_iters):
        vec = np.random.randn(dim)
        vec = vec / np.linalg.norm(vec)
        trajectory.append(vec)
    
    return trajectory

def create_slow_convergence_trajectory(dim=768, num_iters=50):
    """Create trajectory that converges very slowly (shouldn't converge in num_iters)."""
    base = np.random.randn(dim)
    base = base / np.linalg.norm(base)
    
    trajectory = [base]
    for i in range(1, num_iters):
        # Add constant moderate noise
        noise_scale = 0.15
        noise = np.random.randn(dim) * noise_scale
        new_vec = trajectory[-1] + noise
        new_vec = new_vec / np.linalg.norm(new_vec)
        trajectory.append(new_vec)
    
    return trajectory

def test_convergence_detection():
    """Run test cases."""
    print("="*70)
    print("CONVERGENCE DETECTION VERIFICATION")
    print("="*70)
    print("\nTesting with window_size=3, threshold=0.95")
    
    tests = [
        ("Fixed Point (should converge)", create_fixed_point_trajectory(), True, "fixed_point"),
        ("Limit Cycle (should converge)", create_limit_cycle_trajectory(), True, "limit_cycle_2"),
        ("Random Walk (should NOT converge)", create_diverging_trajectory(), False, "none"),
        ("Slow Convergence (should NOT converge)", create_slow_convergence_trajectory(), False, "none"),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, trajectory, expected_converge, expected_type in tests:
        print("\n" + "-"*70)
        print(f"TEST: {test_name}")
        print("-"*70)
        
        # Show similarities
        print("Similarities to previous iteration:")
        for i in range(1, len(trajectory)):
            sim = cosine_similarity(trajectory[i], trajectory[i-1])
            print(f"  iter {i-1}→{i}: {sim:.4f}")
        
        # Detect convergence
        converged, conv_iter, attr_type = detect_convergence(trajectory)
        
        print(f"\nResult:")
        print(f"  Converged: {converged}")
        print(f"  Type: {attr_type}")
        print(f"  Iteration: {conv_iter}")
        
        # Check if matches expected
        if converged == expected_converge:
            if not converged or attr_type == expected_type:
                print("  ✓ PASS")
                passed += 1
            else:
                print(f"  ✗ FAIL (wrong type: expected {expected_type}, got {attr_type})")
                failed += 1
        else:
            print(f"  ✗ FAIL (expected converged={expected_converge}, got {converged})")
            failed += 1
    
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n✓ All tests passed! Convergence detection is working correctly.")
    else:
        print("\n✗ Some tests failed. Check the algorithm implementation.")

def show_math_examples():
    """Show step-by-step math for understanding."""
    print("\n" + "="*70)
    print("STEP-BY-STEP MATH EXAMPLES")
    print("="*70)
    
    print("\nExample 1: Fixed Point Detection")
    print("-"*70)
    
    # Simple 3D vectors for clarity
    embeddings = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.95, 0.05, 0.0]),
        np.array([0.96, 0.04, 0.0]),
        np.array([0.97, 0.03, 0.0]),
        np.array([0.98, 0.02, 0.0]),
    ]
    
    # Normalize
    embeddings = [e / np.linalg.norm(e) for e in embeddings]
    
    print("Embeddings (3D for clarity):")
    for i, e in enumerate(embeddings):
        print(f"  iter {i}: [{e[0]:.3f}, {e[1]:.3f}, {e[2]:.3f}]")
    
    print("\nSimilarities:")
    for i in range(1, len(embeddings)):
        sim = cosine_similarity(embeddings[i], embeddings[i-1])
        print(f"  sim(e{i}, e{i-1}) = {sim:.4f}")
    
    print("\nChecking last 3 iterations (window=3):")
    recent_sims = []
    for i in range(2, 5):  # iterations 2,3,4 (last 3)
        sim = cosine_similarity(embeddings[i], embeddings[i-1])
        recent_sims.append(sim)
        print(f"  sim(e{i}, e{i-1}) = {sim:.4f}")
    
    mean_sim = np.mean(recent_sims)
    print(f"\nMean of recent sims: {mean_sim:.4f}")
    print(f"Threshold: 0.95")
    
    if mean_sim > 0.95:
        print("✓ Mean > threshold → CONVERGED to fixed point")
    else:
        print("✗ Mean ≤ threshold → NOT converged")
    
    print("\n" + "="*70)
    print("Example 2: Limit Cycle Detection")
    print("-"*70)
    
    # Two states oscillating
    state_a = np.array([1.0, 0.0, 0.0])
    state_b = np.array([0.0, 1.0, 0.0])
    
    cycle_embeddings = [state_a, state_b, state_a, state_b, state_a, state_b, state_a]
    cycle_embeddings = [e / np.linalg.norm(e) for e in cycle_embeddings]
    
    print("Embeddings (oscillating):")
    for i, e in enumerate(cycle_embeddings):
        print(f"  iter {i}: [{e[0]:.1f}, {e[1]:.1f}, {e[2]:.1f}]")
    
    print("\nChecking period-2 cycle:")
    print("Compare each iteration with 2 steps back:")
    for i in range(2, len(cycle_embeddings)):
        sim = cosine_similarity(cycle_embeddings[i], cycle_embeddings[i-2])
        print(f"  sim(e{i}, e{i-2}) = {sim:.4f}")
    
    print("\nLast few period-2 similarities:")
    cycle_sims = []
    for i in range(3, 7, 2):  # 3, 5
        sim = cosine_similarity(cycle_embeddings[i], cycle_embeddings[i-2])
        cycle_sims.append(sim)
        print(f"  sim(e{i}, e{i-2}) = {sim:.4f}")
    
    mean_cycle = np.mean(cycle_sims)
    print(f"\nMean of cycle sims: {mean_cycle:.4f}")
    
    if mean_cycle > 0.95:
        print("✓ Mean > threshold → CONVERGED to limit cycle (period 2)")

if __name__ == "__main__":
    test_convergence_detection()
    show_math_examples()
    
    print("\n" + "="*70)
    print("VERIFICATION COMPLETE")
    print("="*70)
    print("\nYou now understand:")
    print("  • Fixed point: System stops changing (x_t = x_{t-1})")
    print("  • Limit cycle: System oscillates (x_t = x_{t-2})")
    print("  • How the detection algorithm works")
    print("  • Why window size and threshold matter")
    print("\nNext: Apply this understanding to your real data!")