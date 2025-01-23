# precalculate_solutions.py

import numpy as np
from numerical_solver import solve_numerically, save_numerical_solution, tesla_to_au
import time
import multiprocessing
from tqdm import tqdm


def calculate_for_B_field(B_tesla):
    """Berechnet die numerische Lösung für ein bestimmtes B-Feld"""
    try:
        B_au = tesla_to_au(B_tesla)
        print(f"Berechne für B = {B_tesla} T ({B_au:.6e} a.u.)")

        states = solve_numerically(B_au)
        metadata = {
            'calculation_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'grid_points': 75,
            'states_calculated': len(states),
            'B_tesla': B_tesla,
            'B_au': B_au
        }
        save_numerical_solution(B_tesla, states, metadata)
        return True
    except Exception as e:
        print(f"Fehler bei B = {B_tesla} T: {str(e)}")
        return False


def main():
    B_fields_tesla = np.linspace(0, 10000, 5)  # 0 bis 10 Tesla in 0.5T Schritten

    print("Geplante Berechnungen:")
    for B in B_fields_tesla:
        print(f"B = {B:.3f} T ({tesla_to_au(B):.6e} a.u.)")

    num_cores = multiprocessing.cpu_count()
    print(f"\nNutze {num_cores} CPU-Kerne für die Berechnung")

    with multiprocessing.Pool(num_cores-1) as pool:
        results = list(tqdm(
            pool.imap(calculate_for_B_field, B_fields_tesla),
            total=len(B_fields_tesla),
            desc="Berechne numerische Lösungen"
        ))

    successful = sum(results)
    print(f"\nBerechnung abgeschlossen:")
    print(f"Erfolgreich: {successful}/{len(B_fields_tesla)}")


if __name__ == '__main__':
    main()