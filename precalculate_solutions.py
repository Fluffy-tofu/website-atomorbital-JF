import numpy as np
import time
import argparse
import multiprocessing
from tqdm import tqdm
import os

# Import only what we need from numerical_solver
from numerical_solver import tesla_to_au, save_hamilton_matrix, V_total


def calculate_hamilton_matrix_only(B_tesla):
    """Calculates only the Hamilton matrix for a specific B-field, no eigenstates"""
    try:
        B_au = tesla_to_au(B_tesla)
        print(f"Calculating Hamilton matrix for B = {B_tesla} T ({B_au:.6e} a.u.)")

        # Problem parameters
        n = 100  # grid points
        a = max(20, 40 / np.sqrt(1 + abs(B_au)))  # adaptive box size
        d = a / n  # step size

        # Create grid
        x = np.linspace(-a / 2, a / 2, n)
        y = np.linspace(-a / 2, a / 2, n)

        # Create Hamiltonian
        N = (n - 2) ** 2
        H = np.zeros((N, N))

        # Build Hamiltonian
        for i in range(n - 2):
            for j in range(n - 2):
                idx = i * (n - 2) + j
                xi = x[i + 1]
                yi = y[j + 1]

                # Diagonal term
                H[idx, idx] = 4 + d ** 2 * V_total(xi, yi, B_au)

                # Off-diagonal terms
                if j < n - 3: H[idx, idx + 1] = -1
                if j > 0: H[idx, idx - 1] = -1
                if i < n - 3: H[idx, idx + (n - 2)] = -1
                if i > 0: H[idx, idx - (n - 2)] = -1

        # Save only the Hamilton matrix
        metadata = {
            'calculation_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'grid_points': n,
            'box_size': a,
            'B_tesla': B_tesla,
            'B_au': B_au
        }
        save_hamilton_matrix(B_tesla, H, metadata)

        return True
    except Exception as e:
        print(f"Error for B = {B_tesla} T: {str(e)}")
        return False


def calculate_full_solution(B_tesla):
    """Calculates the full solution for a specific B-field (both matrix and eigenstates)"""
    try:
        # Import the full solver only when needed to avoid circular imports
        from numerical_solver import solve_numerically, save_numerical_solution

        B_au = tesla_to_au(B_tesla)
        print(f"Calculating full solution for B = {B_tesla} T ({B_au:.6e} a.u.)")

        # This will already save the Hamilton matrix
        states = solve_numerically(B_au)

        # Also save the states
        metadata = {
            'calculation_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'grid_points': 100,
            'states_calculated': len(states),
            'B_tesla': B_tesla,
            'B_au': B_au
        }
        save_numerical_solution(B_tesla, states, metadata)
        return True
    except Exception as e:
        print(f"Error for B = {B_tesla} T: {str(e)}")
        return False


def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(
        description='Precalculate Hamilton matrices and/or solutions for specific B fields')

    # Options for specifying B fields
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--range', nargs=3, type=float, metavar=('START', 'STOP', 'STEPS'),
                       help='Range of B-field values to calculate (start, stop, number of steps)')
    group.add_argument('--values', nargs='+', type=float,
                       help='Specific B-field values in Tesla to calculate')

    # Option to calculate only matrices or full solutions
    parser.add_argument('--matrix-only', action='store_true',
                        help='Calculate only the Hamilton matrices, not the eigenstates')

    # Number of CPU cores to use
    parser.add_argument('--cores', type=int, default=max(1, multiprocessing.cpu_count() - 1),
                        help='Number of CPU cores to use for calculation')

    args = parser.parse_args()

    # Determine B field values to calculate
    if args.range:
        start, stop, steps = args.range
        B_fields_tesla = np.linspace(start, stop, int(steps))
    else:
        B_fields_tesla = np.array(args.values)

    # Choose the calculation function based on the mode
    calc_function = calculate_hamilton_matrix_only if args.matrix_only else calculate_full_solution

    print(f"Calculation mode: {'Hamilton matrix only' if args.matrix_only else 'Full solution'}")
    print(f"B-field values to calculate:")
    for B in B_fields_tesla:
        print(f"  B = {B:.3f} T ({tesla_to_au(B):.6e} a.u.)")

    print(f"\nUsing {args.cores} CPU cores for calculation")

    # Create output directory if it doesn't exist
    os.makedirs('numerical_data', exist_ok=True)

    # Use multiprocessing pool to calculate in parallel
    with multiprocessing.Pool(args.cores) as pool:
        results = list(tqdm(
            pool.imap(calc_function, B_fields_tesla),
            total=len(B_fields_tesla),
            desc="Calculating matrices"
        ))

    successful = sum(results)
    print(f"\nCalculation completed:")
    print(f"Successfully calculated: {successful}/{len(B_fields_tesla)}")


if __name__ == '__main__':
    main()