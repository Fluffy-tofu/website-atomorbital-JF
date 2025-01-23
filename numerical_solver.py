# numerical_solver.py

import os
import json
import numpy as np
from pathlib import Path
import datetime

# Konstanten
NUMERICAL_DATA_DIR = 'numerical_data'
TESLA_TO_AU = 4.25438e-6  # Umrechnungsfaktor Tesla zu atomaren Einheiten

if not os.path.exists(NUMERICAL_DATA_DIR):
    os.makedirs(NUMERICAL_DATA_DIR)


def tesla_to_au(B_tesla):
    """Wandelt Tesla in atomare Einheiten um"""
    return B_tesla * TESLA_TO_AU


def au_to_tesla(B_au):
    """Wandelt atomare Einheiten in Tesla um"""
    return B_au / TESLA_TO_AU


def V_total(x, y, B):
    """
    Total potential including:
    - Coulomb potential
    - Paramagnetic term (∝ B)
    - Diamagnetic term (∝ B²)
    """
    r = np.sqrt(x ** 2 + y ** 2)

    # Coulomb potential with soft core to avoid singularity
    V_coulomb = -1 / np.sqrt(r ** 2 + 0.1)

    # Magnetic terms
    # Lz term (paramagnetic) - scaled with B
    V_para = 0.5 * B * (x * y - y * x)

    # Diamagnetic term - scaled with B²
    V_dia = (B ** 2 / 8) * (x ** 2 + y ** 2)

    return V_coulomb + V_para + V_dia


def solve_numerically(B, num_states=8):
    """Numerische Lösung der Schrödinger-Gleichung"""
    # Problem parameters
    n = 100  # grid points
    a = max(20, 40 / np.sqrt(1 + abs(B)))  # adaptive box size
    d = a / n  # step size

    # Create grid
    x = np.linspace(-a / 2, a / 2, n)
    y = np.linspace(-a / 2, a / 2, n)
    X, Y = np.meshgrid(x, y)

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
            H[idx, idx] = 4 + d ** 2 * V_total(xi, yi, B)

            # Off-diagonal terms
            if j < n - 3: H[idx, idx + 1] = -1
            if j > 0: H[idx, idx - 1] = -1
            if i < n - 3: H[idx, idx + (n - 2)] = -1
            if i > 0: H[idx, idx - (n - 2)] = -1

    # Solve eigenvalue problem
    eigenvalues, eigenvectors = np.linalg.eigh(H)

    # Process results
    results = []
    for i in range(min(num_states, len(eigenvalues))):
        # Reshape eigenstate
        psi = eigenvectors[:, i].reshape((n - 2, n - 2))

        # Calculate quantum numbers (approximate)
        nodes_x = np.sum(np.diff(np.signbit(psi), axis=0) != 0)
        nodes_y = np.sum(np.diff(np.signbit(psi), axis=1) != 0)
        n_approx = max(nodes_x, nodes_y) + 1
        l_approx = abs(nodes_x - nodes_y)
        m_approx = int(np.round(np.angle(np.mean(psi)) / np.pi))

        results.append({
            'energy': float(eigenvalues[i] / (2 * d * d)),
            'state': psi,
            'quantum_numbers': {
                'n': n_approx,
                'l': l_approx,
                'm': m_approx
            }
        })

    return results


def save_numerical_solution(B_tesla, states, metadata=None):
    """Speichert eine vorberechnete numerische Lösung."""
    filename = f"numerical_solution_B{B_tesla:.3f}T.json"
    filepath = os.path.join(NUMERICAL_DATA_DIR, filename)

    data = {
        'B_tesla': B_tesla,
        'B_au': tesla_to_au(B_tesla),
        'timestamp': str(datetime.datetime.now()),
        'metadata': metadata or {},
        'states': []
    }

    for state in states:
        saved_state = {
            'energy': float(state['energy']),
            'wavefunction': state['state'].tolist(),
            'quantum_numbers': {
                'n': int(state['quantum_numbers']['n']),
                'l': int(state['quantum_numbers']['l']),
                'm': int(state['quantum_numbers']['m'])
            }
        }
        data['states'].append(saved_state)

    with open(filepath, 'w') as f:
        json.dump(data, f)

    print(f"Numerische Lösung gespeichert in: {filepath}")


def load_numerical_solution(B_tesla, tolerance=0.1):
    """Lädt eine vorberechnete numerische Lösung."""
    best_match = None
    smallest_diff = float('inf')

    for file in Path(NUMERICAL_DATA_DIR).glob('numerical_solution_B*.json'):
        try:
            file_B = float(file.stem.split('B')[1].replace('T', ''))
            diff = abs(file_B - B_tesla)

            if diff < smallest_diff and diff <= tolerance:
                smallest_diff = diff
                best_match = file
        except:
            continue

    if best_match is None:
        return None

    with open(best_match, 'r') as f:
        data = json.load(f)

    for state in data['states']:
        state['state'] = np.array(state['wavefunction'])
        del state['wavefunction']

    return data
