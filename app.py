from flask import Flask, render_template, jsonify, request, Response
import numpy as np
from scipy.special import factorial, lpmv, genlaguerre
from helper_functions import GeneralFunctions
import math
import json
import os
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from io import BytesIO
import base64
from pathlib import Path
import multiprocessing
import psutil
from scipy.constants import e, hbar
from numerical_solver import load_hamilton_matrix, save_hamilton_matrix, tesla_to_au, V_total, load_numerical_solution

app = Flask(__name__)

# hbar = 1.054571817e-34
m_e = 9.1093837015e-31
# e = 1.602176634e-19
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


# @app.before_first_request
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

        n = int(data['n'])
        l = int(data['l'])
        m = int(data['m'])

        # Retrieve and enforce limits on the number of points
        num_points = int(data.get('points', 4000))
        num_points = min(max(num_points, 1000), 80000)  # Clamp to [1000, 80000]

        scatter = float(data.get('scatter', 10)) / 30.0
        perturbation = data.get('perturbation', 'none')
        B = float(data.get('field', 0))
        threshold = float(data.get('threshold', 0.01))

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

        # Filter points by threshold
        filtered_points = [p for p in points if p['density'] >= threshold]

        print(f"Berechnung erfolgreich. Anzahl der Punkte: {len(filtered_points)} (von {len(points)})")
        return jsonify({'points': filtered_points})

    except Exception as e:
        print(f"Fehler in calculate: {str(e)}")
        return jsonify({'error': str(e)}), 500


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


@app.route('/numerical')
def numerical_simulation():
    return render_template('numerical_simulation.html')


@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/equation')
def equation():
    return render_template('matrix_eigenwert.html')


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


NUMERICAL_DATA_DIR = 'numerical_data'
TESLA_TO_AU = 4.25438e-6  # Umrechnungsfaktor Tesla zu atomaren Einheiten

if not os.path.exists(NUMERICAL_DATA_DIR):
    os.makedirs(NUMERICAL_DATA_DIR)


@app.route('/hamilton_matrix')
def hamilton_matrix_visualization():
    """Serves the HTML page for Hamilton matrix visualization"""
    return render_template('hamilton_matrix.html')


@app.route('/get_hamilton_matrix', methods=['POST'])
def get_hamilton_matrix():
    """API endpoint to get Hamilton matrix data for visualization"""
    try:
        data = request.json
        B_tesla = float(data['B'])
        grid_size = int(data.get('gridSize', 75))  # Default to 75 if not specified

        # Try to load a pre-calculated Hamilton matrix
        matrix_data = load_hamilton_matrix(B_tesla)

        if matrix_data is None:
            # If no pre-calculated matrix found, calculate in real-time
            B_au = tesla_to_au(B_tesla)
            # We need to recreate the matrix here instead of calling solve_numerically
            # to avoid redundant eigenvalue calculation
            n = grid_size  # grid points
            a = max(20, 40 / np.sqrt(1 + abs(B_au)))  # adaptive box size
            d = a / n  # step size

            x = np.linspace(-a / 2, a / 2, n)
            y = np.linspace(-a / 2, a / 2, n)

            N = (n - 2) ** 2
            H = np.zeros((N, N))

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

            matrix_data = {
                'H': H,
                'B_tesla': B_tesla,
                'B_au': B_au,
                'metadata': {'grid_size': n, 'box_size': a}
            }

            # Save for future use
            save_hamilton_matrix(B_tesla, H, metadata=matrix_data['metadata'])

        # Convert to a sparse representation for efficient transmission
        H = matrix_data['H']
        non_zero_idxs = np.abs(H) > 1e-6  # threshold for "non-zero"
        rows, cols = np.where(non_zero_idxs)
        values = H[non_zero_idxs]

        # Convert all NumPy types to Python native types for JSON serialization
        response = {
            'rows': rows.tolist(),
            'cols': cols.tolist(),
            'values': values.tolist(),
            'shape': [int(dim) for dim in H.shape],
            'B_tesla': float(matrix_data['B_tesla']),
            'source': 'cached' if 'timestamp' in matrix_data else 'realtime',
            'metadata': matrix_data.get('metadata', {})
        }

        return jsonify(response)

    except Exception as e:
        print(f"Fehler in get_hamilton_matrix: {str(e)}")
        import traceback
        traceback.print_exc()  # This will print the full traceback
        return jsonify({'error': str(e)}), 500


@app.route('/calculate_numerical', methods=['POST'])
def calculate_numerical():
    try:
        data = request.json
        B_tesla = float(data['B'])
        selected_state = int(data.get('selectedState', 0))

        # Versuche zuerst, eine vorberechnete Lösung zu laden
        numerical_data = load_numerical_solution(B_tesla)

        if numerical_data is None:
            # Wenn keine vorberechnete Lösung gefunden wurde,
            # verwende die Echtzeit-Berechnung mit B-Feld in a.u.
            # We would call solve_numerically here from numerical_solver,
            # but for simplicity in this integrated example, let's assume it exists
            from numerical_solver import solve_numerically
            states = solve_numerically(tesla_to_au(B_tesla))
        else:
            states = numerical_data['states']

        # Wähle den gewünschten Zustand
        if selected_state < len(states):
            state = states[selected_state]

            response = {
                'energy': float(state['energy']),
                'state': state['state'].flatten().tolist(),
                'quantum_numbers': convert_numpy_types(state['quantum_numbers']),
                'source': 'cached' if numerical_data else 'realtime',
                'B_tesla': B_tesla,
                'B_au': tesla_to_au(B_tesla)
            }

            return jsonify(response)
        else:
            return jsonify({'error': 'Ausgewählter Zustand nicht verfügbar'}), 400

    except Exception as e:
        print(f"Fehler in calculate_numerical: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/available_b_fields')
def get_available_b_fields():
    try:
        available_fields = []
        for file in Path(NUMERICAL_DATA_DIR).glob('numerical_solution_B*.json'):
            try:
                B_tesla = float(file.stem.split('B')[1].replace('T', ''))
                available_fields.append(B_tesla)
            except:
                continue

        # Also include Hamilton matrix files
        for file in Path(NUMERICAL_DATA_DIR).glob('hamilton_matrix_B*.npz'):
            try:
                B_tesla = float(file.stem.split('B')[1].replace('T', ''))
                if B_tesla not in available_fields:
                    available_fields.append(B_tesla)
            except:
                continue

        return jsonify({
            'fields': sorted(available_fields),
            'states': 8  # Angepasst an die solve_numerically Funktion
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/bar_chart')
def get_bar_chart():
    # set up the figure and Axes
    fig = plt.figure(figsize=(8, 3))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    # fake data
    x = np.arange(4)
    y = np.arange(5)
    xx, yy = np.meshgrid(x, y)
    x, y = xx.ravel(), yy.ravel()

    top = x + y
    bottom = np.zeros_like(top)
    width = depth = 1

    ax1.bar3d(x, y, bottom, width, depth, top, shade=True)
    ax1.set_title('Shaded')

    ax2.bar3d(x, y, bottom, width, depth, top, shade=False)
    ax2.set_title('Not Shaded')

    # Save plot to a BytesIO object
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    # Return the image
    return Response(buffer.getvalue(), mimetype='image/png')


@app.route('/api/bson_matrix/<path:filename>')
def get_bson_matrix_data(filename):
    try:
        # Load the BSON file
        filepath = os.path.join('converted_data', filename)
        with open(filepath, 'rb') as f:
            bson_data = f.read()

        # Return as binary response
        return Response(bson_data, mimetype='application/bson')
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/matrix/<path:filename>')
def get_matrix_data(filename):
    try:
        # Load the NPZ file
        filepath = os.path.join('numerical_data', filename)
        data = np.load(filepath)

        # Extract the matrix using the correct key 'H' instead of 'matrix'
        matrix = data['H']

        # Get additional metadata that was saved with the file
        B_tesla = float(data['B_tesla'])
        B_au = float(data['B_au'])
        timestamp = str(data['timestamp'])

        # Handle the metadata which was saved as a JSON string
        metadata = json.loads(str(data['metadata']))

        # Convert to Python list for JSON serialization
        matrix_list = matrix.tolist()

        return jsonify({
            'filename': filename,
            'size': matrix.shape,
            'matrix': matrix_list,
            'B_tesla': B_tesla,
            'B_au': B_au,
            'timestamp': timestamp,
            'metadata': metadata
        })
    except Exception as e:
        print(f"Error loading matrix: {str(e)}")  # Log the error for debugging
        return jsonify({'error': str(e)}), 500


@app.route("/api/eigenwerte/<path:filename>")
def energy_values(filename):
    filepath = os.path.join('numerical_data', filename)

    try:
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Process energy values to extract plain numbers
        energy_values = []
        raw_values = []

        # Figure out where the energy values are in the structure
        if isinstance(data, dict):
            if "states" in data:
                if isinstance(data["states"], dict) and "energy" in data["states"]:
                    raw_values = data["states"]["energy"]
                elif isinstance(data["states"], list):
                    raw_values = data["states"]
            elif "energy" in data:
                raw_values = data["energy"]

        # Convert complex structures to simple numbers if needed
        for val in raw_values:
            if isinstance(val, (int, float)):
                energy_values.append(val)
            elif isinstance(val, dict) and "value" in val:
                energy_values.append(val["value"])
            elif isinstance(val, dict) and len(val) > 0:
                # Take the first numeric value we find
                for k, v in val.items():
                    if isinstance(v, (int, float)):
                        energy_values.append(v)
                        break
            else:
                # If all else fails, add a placeholder
                energy_values.append(None)

        # Get B_tesla value with fallback
        B_tesla = data.get("B_tesla", float(filename.split("B")[1].replace("T.json", "")))

        return jsonify({
            'energy': energy_values,
            'B_tesla': B_tesla
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route("/eigenwerte")
def eigenwerte():
    return render_template("eigenwerte.html")

if __name__ == '__main__':
    initialize()
    app.run(debug=True)