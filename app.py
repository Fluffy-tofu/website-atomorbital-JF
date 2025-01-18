from flask import Flask, render_template, jsonify, request
import numpy as np
from scipy.special import factorial, lpmv, genlaguerre
from helper_functions import GeneralFunctions
import math
import json
import multiprocessing
import psutil
from scipy.constants import e, hbar

app = Flask(__name__)

#hbar = 1.054571817e-34
m_e = 9.1093837015e-31
#e = 1.602176634e-19
mu_B = 9.2740100783e-24
a0 = 5.29177e-11

# Optimize for numerical computations
def configure_numpy():
    import numpy as np
    # Use multiple threads for numerical computations
    np.show_config()
    # Set number of threads based on available CPU cores
    threads = multiprocessing.cpu_count()
    return threads

def monitor_resources():
    cpu_percent = psutil.cpu_percent()
    memory_info = psutil.Process().memory_info()
    return {
        'cpu_percent': cpu_percent,
        'memory_used_mb': memory_info.rss / 1024 / 1024
    }

#@app.before_first_request
def initialize():
    global NUM_THREADS
    NUM_THREADS = configure_numpy()
    app.logger.info(f"Configured with {NUM_THREADS} threads")

@app.route('/health')
def health():
    return monitor_resources()

class Constants:
    """Physikalische Konstanten"""
    hbar = 1.054571817e-34
    m_e = 9.1093837015e-31
    e = 1.602176634e-19
    mu_B = 9.2740100783e-24
    a0 = 5.29177e-11  # Bohr-Radius


def Y_lm(l, m, theta, phi):
    """
    Kugelflächenfunktion für beliebige l und m.
    """
    if l < 0 or abs(m) > l:
        raise ValueError("Ungültige Werte für l oder m")

    # Normierungsfaktor
    norm = np.sqrt((2 * l + 1) / (4 * np.pi) * factorial(l - abs(m)) / factorial(l + abs(m)))

    # Legendre-Polynome
    P_lm = lpmv(abs(m), l, np.cos(theta))

    if m < 0:
        return np.sqrt(2) * norm * P_lm * np.sin(abs(m) * phi)
    elif m > 0:
        return np.sqrt(2) * norm * P_lm * np.cos(m * phi)
    else:
        return norm * P_lm


def R_nl(n, l, r, Z=1):
    """
    Radialer Teil der Wellenfunktion
    """
    try:
        rho = (2 * Z * r) / n
        norm = np.sqrt((2 * Z / n) ** 3 * math.factorial(n - l - 1) /
                       (2 * n * math.factorial(n + l)))
        laguerre_poly = genlaguerre(n - l - 1, 2 * l + 1)(rho)
        radial_part = norm * np.exp(-rho / 2) * (rho ** l) * laguerre_poly

        # Handhabung von numerischen Problemen
        radial_part = np.nan_to_num(radial_part, nan=0.0, posinf=0.0, neginf=0.0)
        return radial_part
    except Exception as e:
        print(f"Warning: Error in radial function for n={n}, l={l}: {str(e)}")
        return np.ones_like(r) * 0.1


def calculate_matrix_element(psi_k, psi_0, B, r, theta, phi, m):
    """Berechnet das Matrixelement für die Störungstheorie"""
    # Zeeman Term
    zeeman = Constants.mu_B * B * m * np.conjugate(psi_k) * psi_0

    # Diamagnetischer Term
    r_perp_squared = r ** 2 * np.sin(theta) ** 2
    dia = (Constants.e ** 2 * B ** 2 / (8 * Constants.m_e)) * np.conjugate(psi_k) * r_perp_squared * psi_0

    return zeeman + dia


def first_order_correction(n, l, m, B, R, Theta, Phi):
    """Störungstheorie erster Ordnung"""
    # Ungestörte Wellenfunktion
    psi_0 = R_nl(n, l, R) * Y_lm(l, m, Theta, Phi)

    # Störterme
    H_para = Constants.mu_B * B * m
    r_perp_squared = R ** 2 * np.sin(Theta) ** 2
    H_dia = (Constants.e ** 2 * B ** 2 / (8 * Constants.m_e)) * r_perp_squared

    # Gestörte Wellenfunktion
    psi = psi_0 * (1 + H_para + H_dia)

    # Normierung
    dV = R ** 2 * np.sin(Theta)
    norm = np.sqrt(np.sum(np.abs(psi) ** 2 * dV))
    return psi / (norm + 1e-10)  # Kleine Konstante zur Vermeidung von Division durch 0


def second_order_correction(n, l, m, B, R, Theta, Phi):
    """Vereinfachte Störungstheorie zweiter Ordnung"""
    # Ungestörte Wellenfunktion
    psi_0 = R_nl(n, l, R) * Y_lm(l, m, Theta, Phi)

    # Störterme zweiter Ordnung
    H_para_2 = (Constants.mu_B * B * m) ** 2 / 2
    r_perp_squared = R ** 2 * np.sin(Theta) ** 2
    H_dia_2 = ((Constants.e ** 2 * B ** 2) / (8 * Constants.m_e)) ** 2 * r_perp_squared ** 2 / 2
    H_cross = Constants.mu_B * B * m * (Constants.e ** 2 * B ** 2 / (8 * Constants.m_e)) * r_perp_squared

    # Gestörte Wellenfunktion
    psi = psi_0 * (1 + H_para_2 + H_dia_2 + H_cross)

    # Normierung
    dV = R ** 2 * np.sin(Theta)
    norm = np.sqrt(np.sum(np.abs(psi) ** 2 * dV))
    return psi / (norm + 1e-10)


def calculate_probability_density(n, l, m, num_points=4000, scatter_factor=0.2, resolution=30, perturbation='none',
                                  B=0):
    """Berechnet die Wahrscheinlichkeitsdichte mit optionaler Störungstheorie"""

    visual_dict = {}

    functions = GeneralFunctions(visual_dict=visual_dict)
    try:
        # Erstelle Gitter
        r = np.linspace(0, 15, resolution)
        theta = np.linspace(0, np.pi, resolution)
        phi = np.linspace(0, 2 * np.pi, resolution)
        R, Theta, Phi = np.meshgrid(r, theta, phi, indexing='ij')

        # Wähle Wellenfunktion basierend auf Störungstheorie
        if perturbation == 'none':
            psi = R_nl(n, l, R) * Y_lm(l, m, Theta, Phi)
        elif perturbation == 'first':
            psi = functions.first_order_correction(n, l, m, B, R, Theta, Phi)
            functions.validate_implementation()
        else:  # second
            psi = functions.total_correction_up_to_second_order(n, l, m, B, R, Theta, Phi)

        # Berechne Wahrscheinlichkeitsdichte
        density = np.abs(psi) ** 2 * R ** 2 * np.sin(Theta)
        density = density / (np.max(density) + 1e-10)

        # Konvertiere zu kartesischen Koordinaten
        X = R * np.sin(Theta) * np.cos(Phi)
        Y = R * np.sin(Theta) * np.sin(Phi)
        Z = R * np.cos(Theta)

        # Wähle Punkte basierend auf Wahrscheinlichkeitsdichte
        points = []
        total_points = X.size
        probabilities = density.flatten()
        probabilities = probabilities / np.sum(probabilities)

        indices = np.random.choice(total_points, size=num_points, p=probabilities)

        for idx in indices:
            scatter = (np.random.random(3) - 0.5) * scatter_factor
            points.append({
                'x': float(X.flat[idx] + scatter[0]),
                'y': float(Y.flat[idx] + scatter[1]),
                'z': float(Z.flat[idx] + scatter[2]),
                'density': float(density.flat[idx])
            })

        return points
    except Exception as e:
        print(f"Fehler in calculate_probability_density: {str(e)}")
        raise


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/calculate', methods=['POST'])
def calculate():
    try:
        data = request.json
        print("\n=== Neue Berechnung ===")
        print(f"Erhaltene Daten: {data}")

        data = request.json
        n = int(data['n'])
        l = int(data['l'])
        m = int(data['m'])

        num_points = int(data.get('points', 4000))
        scatter = float(data.get('scatter', 10)) / 30.0
        perturbation = data.get('perturbation', 'none')
        B = float(data.get('field', 0))
        threshold = float(data.get('threshold', 0.01))

        if perturbation == 'numerical':
            # Verwende numerische Lösung
            states = solve_numerically(B)
            points = []

            # Konvertiere den ausgewählten Zustand in Punktwolke
            for state in states:
                psi = state['state']
                qn = state['quantum_numbers']

                # Generiere Punktwolke basierend auf |ψ|²
                prob = np.abs(psi) ** 2
                prob = prob / np.max(prob)

                # Erstelle 3D-Punktwolke
                x = []
                y = []
                z = []
                densities = []

                for i in range(len(psi)):
                    for j in range(len(psi)):
                        if prob[i, j] > 0.01:  # Schwellwert
                            x.append(float(i))
                            y.append(float(j))
                            z.append(0.0)  # Für 2D-Darstellung
                            densities.append(float(prob[i, j]))

                points.append({
                    'points': list(zip(x, y, z)),
                    'densities': densities,
                    'energy': state['energy'],
                    'quantum_numbers': qn
                })

            return jsonify({
                'numerical_states': points,
                'method': 'numerical'
            })

        else:

            # Validierung
            if l >= n:
                return jsonify({'error': 'l muss kleiner als n sein'}), 400
            if abs(m) > l:
                return jsonify({'error': '|m| muss kleiner oder gleich l sein'}), 400

            points = calculate_probability_density(
                n, l, m,
                num_points=num_points,
                scatter_factor=scatter,
                perturbation=perturbation,
                B=B
            )

            # Filtere Punkte nach Schwellwert
            filtered_points = [p for p in points if p['density'] >= threshold]

            print(f"Berechnung erfolgreich. Anzahl der Punkte: {len(filtered_points)} (von {len(points)})")
            return jsonify({'points': filtered_points})

    except Exception as e:
        print(f"Fehler in calculate: {str(e)}")
        return jsonify({'error': str(e)}), 500

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

    # Diamagnetic term - scaled with B
    V_dia = (B ** 2 / 8) * (x ** 2 + y ** 2)

    return V_coulomb + V_para + V_dia

def solve_numerically(B, num_states=5):
    """Numerische Lösung der Schrödinger-Gleichung"""
    # Problem parameters
    n = 75  # grid points
    a = max(20, 40 / np.sqrt(1 + B))  # adaptive box size
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


# Diese Hilfsfunktion am Anfang der app.py hinzufügen
def convert_numpy_types(obj):
    """Konvertiert NumPy-Datentypen in Python-native Typen"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    return obj





# Füge diese neue Route zur app.py hinzu

@app.route('/numerical')
def numerical_simulation():
    return render_template('numerical_simulation.html')


@app.route('/calculate_numerical', methods=['POST'])
def calculate_numerical():
    try:
        data = request.json
        B = float(data['B'])
        grid_points = int(data.get('gridPoints', 75))
        selected_state = int(data.get('selectedState', 0))

        # Berechne numerische Lösung
        states = solve_numerically(B, grid_points)

        # Wähle den gewünschten Zustand
        if selected_state < len(states):
            state = states[selected_state]

            # Konvertiere numpy arrays und Datentypen zu Python-nativen Typen
            response = {
                'energy': float(state['energy']),
                'state': state['state'].flatten().tolist(),
                'quantum_numbers': convert_numpy_types(state['quantum_numbers'])
            }

            return jsonify(response)
        else:
            return jsonify({'error': 'Ausgewählter Zustand nicht verfügbar'}), 400

    except Exception as e:
        print(f"Fehler in calculate_numerical: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')


@app.route('/calculate_energy', methods=['POST'])
def calculate_energy():
    try:
        data = request.json
        n = int(data['n'])
        l = int(data['l'])
        m = int(data['m'])
        B = float(data['field'])

        # Initialize GeneralFunctions
        functions = GeneralFunctions(visual_dict={})

        # Calculate paramagnetic contribution (linear with B)
        para_contribution = -B * m * 5.788e-5  # μB * B * m in eV

        # Calculate diamagnetic contribution (quadratic with B)
        dia_contribution = B * B * 2.894e-6 * (n ** 2)  # (e²B²/8me)⟨r²⟩ in eV

        # Total energy shift
        total_energy = para_contribution + dia_contribution

        # Generate energy data points for different field strengths
        field_points = np.linspace(0, B, 50)
        energy_data = []

        for field in field_points:
            para = -field * m * 5.788e-5
            dia = field * field * 2.894e-6 * (n ** 2)
            energy_data.append({
                'field': float(field),
                'para': float(para),
                'dia': float(dia),
                'total': float(para + dia)
            })

        return jsonify({
            'energyData': energy_data,
            'current': {
                'paramagnetic': float(para_contribution),
                'diamagnetic': float(dia_contribution),
                'total': float(total_energy)
            }
        })

    except Exception as e:
        print(f"Error in calculate_energy: {str(e)}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    initialize()
    app.run()
