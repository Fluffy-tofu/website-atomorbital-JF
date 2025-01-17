import numpy as np
import scipy.special as sp
from collections import defaultdict
from scipy.special import genlaguerre
from vispy import app, scene
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from itertools import combinations
import math
import scipy.linalg as la
import numpy as np
from scipy.special import lpmv, factorial
import seaborn as sns
from scipy.constants import e, hbar, m_e


mu_B = 5.788e-7 # eV/Tesla
feinstrukturkonstante = 1/137

def Y_lm(l, m, theta, phi):
    """
    Allgemeine Kugelflächenfunktion für beliebige l und m.
    Parameter:
    l : int
        Nebenquantenzahl (l >= 0)
    m : int
        Magnetische Quantenzahl (-l ≤ m ≤ l)
    theta : float oder numpy.ndarray
        Polarwinkel in Radiant (0 <= theta <= pi)
    phi : float oder numpy.ndarray
        Azimuthalwinkel in Radiant (0 <= phi < 2pi)
    Rückgabe:
    Y : complex oder numpy.ndarray
        Wert der Kugelflächenfunktion
    """

    if l < 0 or abs(m) > l:
        raise ValueError("Ungültige Werte für l oder m")

    norm = np.sqrt((2 * l + 1) / (4 * np.pi) * factorial(l - abs(m)) / factorial(l + abs(m)))
    P_lm = lpmv(abs(m), l, np.cos(theta))
    if m < 0:
        return np.sqrt(2) * norm * P_lm * np.sin(abs(m) * phi)
    elif m > 0:
        return np.sqrt(2) * norm * P_lm * np.cos(m * phi)
    else:
        return norm * P_lm



class GeneralFunctions:
    def __init__(self, visual_dict):
        self.visual_dict = visual_dict
        # Physikalische Konstanten
        self.e = e  # Elementarladung
        self.hbar = hbar  # Reduziertes Plancksches Wirkungsquantum
        self.m_e = m_e  # Elektronenmasse
        self.mu_B = mu_B  # Bohrsches Magneton
        self.cached_coordinates = None

    def calculate_orbital_points(self, n, l, m, Z, electron_count, threshold=0.1, num_points=100000, magnetic_field=0):
        """
        Berechne Orbitale-Punkte mit einem mit Schwellenwert
        Parameter:
        -----------
        n, l, m: int
            Quantenzahlen
        Z: int
            Ordnungszahl
        elektronenanzahl: int
            Anzahl der Elektronen im Orbital
        threshold: float
            Mindestwert für die Wahrscheinlichkeitsdichte (0 bis 1)
        num_points: int
            Anzahl der zu generierenden Punkte
        """

        if electron_count == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])

        # Generate coordinates with appropriate scaling
        r, theta, phi = self.generate_grid(n, Z, num_points)

        if magnetic_field != 0:
            density = self.probability_density_magnetic_field(n, l, r, Z, m, theta, phi, magnetic_field)

        else:
            density = self.probability_density(n, l, m, Z, r, theta, phi)

        # Normalize to electron count
        density *= electron_count / (2.0 * np.sum(density) / num_points)

        # Add small offset and normalize
        #density = density + np.max(density) * 0.01
        max_density = np.max(density)
        if max_density > 1e-10:
            density = density / max_density
        else:
            density = np.ones_like(density) * 0.1

        # Safety checks
        density = np.nan_to_num(density, nan=0.1, posinf=1.0, neginf=0.0)
        #density = np.clip(density, 0.01, 1)

        # Apply threshold filter
        mask = (density >= threshold) & (np.random.random(num_points) < (density * 0.5 + 0.1))

        x, y, z = self.convert_cartesian(r, theta, phi)

        return x[mask], y[mask], z[mask], density[mask]

    def probability_density(self, n, l, m, Z, r, theta, phi):
        R_nl = self.radial_function(n, l, r, Z)
        Y_lm = self.spherical_harmonics(l, m, theta, phi)
        density = np.abs(R_nl * Y_lm) ** 2
        return density

    def wave_func(self, n, l, m, Z, r, theta, phi):
        R_nl = self.radial_function(n, l, r, Z)
        Y_lm = self.spherical_harmonics(l, m, theta, phi)
        return R_nl * Y_lm

    def generate_grid(self, n, Z, num_points):
        """Generiert oder verwendet gecachte Koordinaten"""
        if self.cached_coordinates is None:
            r = np.random.exponential(scale=n ** 2 / Z, size=num_points)
            theta = np.arccos(2 * np.random.random(num_points) - 1)
            phi = 2 * np.pi * np.random.random(num_points)
            self.cached_coordinates = (r, theta, phi)
        return self.cached_coordinates

    def clear_cache(self):
        """Löscht den Cache der Koordinaten"""
        self.cached_coordinates = None

    def convert_cartesian(self, r, theta, phi):
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        return x, y, z

    def radial_function(self, n, l, r, Z):
        """Besser radiale Funktion"""
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

    def spherical_harmonics(self, l, m, theta, phi):
        """Sichere Version der Kugelflächenfunktionen"""
        try:
            Y = Y_lm(l, m, theta, phi)
            Y = np.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)
            return Y
        except Exception as e:
            print(f"Warnung: Fehler in Kugelflächenfunktion für l={l}, m={m}: {str(e)}")
            return np.zeros_like(theta)

    def add_orbital(self, n, l, m, Z, electron_count, points_plotted):
        """Visualisierung"""
        x, y, z, density = self.calculate_orbital_points(n, l, m, Z, electron_count,
                                                         threshold=self.visual_dict["prob_threshold"],
                                                         num_points=self.visual_dict["num_points"],
                                                         magnetic_field=self.visual_dict["magnetic_field"])
        points_plotted += len(x)
        if len(x) > 0:
            colors = self.get_density_color(density)

            state = self.visual_dict["state"]
            not_see_inside = self.visual_dict["not_see_inside"]
            blend = self.visual_dict["blend"]
            edges = self.visual_dict["edges"]
            point_size = self.visual_dict["point_size"]

            point_size = 10 * density + 3 if point_size == "dynamic" else point_size
            edge_color = None if edges else colors
            scatter = scene.visuals.Markers()
            scatter.set_gl_state(state, depth_test=not_see_inside, blend=blend)
            scatter.set_data(
                np.column_stack((x, y, z)),
                edge_color=edge_color,
                face_color=colors,
                size=point_size
            )
            self.view.add(scatter)
        return points_plotted

    def get_density_color(self, density):
        red = np.clip(density * 2, 0, 1)
        blue = np.clip(2 - density * 2, 0, 1)
        green = np.clip(1 - np.abs(density - 0.5) * 2, 0, 1)
        alpha = np.clip(density * 0.8 + 0.2, 0, 1)
        return np.column_stack((red, green, blue, alpha))

    def get_density_color_magnetic_field(self, density):
        red = np.clip(2 - density * 2, 0, 1)
        blue = np.clip(density * 2, 0, 1)
        green = np.clip(1 - np.abs(density - 0.5) * 2, 0, 1)
        alpha = np.clip(density * 0.8 + 0.2, 0, 1)
        return np.column_stack((red, green, blue, alpha))

    def calculate_orbitals_energy(self, n, Z):
        energy = -13.6/n**2
        sommerfeld_formular = energy * (Z ** 2 / (1 + feinstrukturkonstante ** 2/n ** 2) ** 2)
        # print(f"Z={Z} || n={n} ---> {sommerfeld_formular}")
        return sommerfeld_formular

    def calculate_orbitals_energy_magneticfield(self, n, m_l, b_field, Z):
        """Erweiterte Version mit diamagnetischem Term"""
        base_energy = self.calculate_orbitals_energy(n, Z)

        # Zeeman Term (orbital)
        zeeman_energy = self.mu_B * b_field * m_l

        # Diamagnetischer Term
        # Erwartungswert von r² für Wasserstoff
        r2_expect = (n ** 2 * (5 * n ** 2 + 1)) * (0.529e-10) ** 2  # in m²
        dia_energy = (self.e ** 2 * b_field ** 2 / (8 * self.m_e)) * r2_expect

        return base_energy + zeeman_energy + dia_energy

    def scaling_factor_magnetic_field(self, E_new, E_base):
        return np.sqrt(abs(E_new)/abs(E_base))

    def first_order_apporoximation_old(self, n, l, r, Z, m, theta, phi, field):
        """Erweiterte Version mit Störungskorrektur"""
        # Basis-Wellenfunktion berechnen
        R_nl = self.radial_function(n, l, r, Z)
        Y_lm = self.spherical_harmonics(l, m, theta, phi)
        psi_0 = R_nl * Y_lm

        # Kartesische Koordinaten für diamagnetischen Term
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)

        # Störungskorrektur
        orbital_term = self.mu_B * field * m
        dia_term = (self.e ** 2 * field ** 2 / (8 * self.m_e)) * (x ** 2 + y ** 2)
        correction = orbital_term + dia_term
        print(dia_term)
        print(orbital_term)

        # Gestörte Wellenfunktion
        psi = psi_0 * (1 + correction)
        return psi

    def first_order_apporoximation(self, n, l, r, Z, m, theta, phi, field):
        """
        Vereinfachte Näherung für den Zeeman-Effekt mit Normierung.

        Parameters:
        -----------
        n, l, m : int
            Quantenzahlen
        r, theta, phi : array
            Kugelkoordinaten
        Z : int
            Kernladungszahl
        field : float
            Magnetfeldstärke
        """
        # Berechne ungestörte Wellenfunktion
        R_nl = self.radial_function(n, l, r, Z)
        Y_lm = self.spherical_harmonics(l, m, theta, phi)
        psi_0 = R_nl * Y_lm

        # Berechne Störterme
        # Paramagnetischer Term
        H_para = self.mu_B * field * m

        # Diamagnetischer Term (In Kugelkoordinaten) (x² + y² = r²sin²(θ))
        r_perp_squared = r * r * np.sin(theta) * np.sin(theta)
        H_dia = (self.e ** 2 * field ** 2 / (8 * self.m_e)) * r_perp_squared

        # Kombinierte gestörte Wellenfunktion (Zähler)
        psi = psi_0 * (1 + H_para + H_dia)

        # Normierung (Nenner)
        # Volumenelement in Kugelkoordinaten
        dV = r ** 2 * np.sin(theta)

        # Berechne Normierungsfaktor
        norm = np.sqrt(np.sum(np.abs(psi) ** 2 * dV))

        # Normiere die Wellenfunktion
        if norm > 0:
            psi = psi / norm

        return psi

    def second_order_correction(self, n, l, m, B, r, theta, phi):
        """
        Second-order perturbation theory implementation für 1D Arrays
        """
        # Convert to atomic units
        a0 = 5.29177e-11
        B_atomic = B * (a0 ** 2 * self.e / self.hbar)

        # Unperturbed wavefunction
        psi_0 = self.wave_func(n, l, m, 1, r, theta, phi)
        E_n = self.calculate_orbitals_energy(n, 1)

        # Initialize correction
        psi_2 = np.zeros_like(psi_0, dtype=complex)

        # Volume element
        dV = r ** 2 * np.sin(theta)

        # Store contributions for debugging
        contributions = []

        # First term: Double transition through ground state
        V_00 = self.calculate_matrix_element(psi_0, psi_0, B_atomic, r, theta, phi, m)
        V_00_integral = np.sum(V_00 * dV)

        # Double sum over intermediate states
        for n_prime in range(max(1, n - 2), n + 3):
            for l_prime in range(max(0, l - 1), min(n_prime, l + 2)):
                for m_prime in range(-l_prime, l_prime + 1):
                    if (n_prime, l_prime, m_prime) == (n, l, m):
                        continue

                    try:
                        # First intermediate state
                        psi_k = self.wave_func(n_prime, l_prime, m_prime, 1, r, theta, phi)
                        E_k = self.calculate_orbitals_energy(n_prime, 1)

                        # First term contribution
                        V_k0 = self.calculate_matrix_element(psi_k, psi_0, B_atomic, r, theta, phi, m)
                        V_k0_integral = np.sum(V_k0 * dV)

                        # Energy denominators
                        delta_E_k = E_k - E_n

                        if abs(delta_E_k) > 1e-10:
                            # First term (transition through ground state)
                            contribution_1 = (V_k0_integral * V_00_integral / (delta_E_k ** 2)) * psi_k

                            # Second term (transitions through intermediate states)
                            contribution_2 = np.zeros_like(psi_k, dtype=complex)

                            for n_m in range(max(1, n - 2), n + 3):
                                for l_m in range(max(0, l - 1), min(n_m, l + 2)):
                                    for m_m in range(-l_m, l_m + 1):
                                        if (n_m, l_m, m_m) == (n, l, m) or (n_m, l_m, m_m) == (
                                        n_prime, l_prime, m_prime):
                                            continue

                                        try:
                                            # Second intermediate state
                                            psi_m = self.wave_func(n_m, l_m, m_m, 1, r, theta, phi)
                                            E_m = self.calculate_orbitals_energy(n_m, 1)

                                            V_km = self.calculate_matrix_element(psi_k, psi_m, B_atomic, r, theta, phi, m)
                                            V_m0 = self.calculate_matrix_element(psi_m, psi_0, B_atomic, r, theta, phi, m)

                                            V_km_integral = np.sum(V_km * dV)
                                            V_m0_integral = np.sum(V_m0 * dV)

                                            delta_E_m = E_m - E_n

                                            if abs(delta_E_m) > 1e-10:
                                                energy_denominator = delta_E_k * delta_E_m
                                                # Prüfen Sie das Vorzeichen des Energienenners
                                                sign = np.sign(energy_denominator)
                                                contribution_2 += sign * (V_km_integral * V_m0_integral /
                                                                          np.abs(energy_denominator)) * psi_k

                                        except Exception as e:
                                            print(f"Exception for second state ({n_m},{l_m},{m_m}): {str(e)}")
                                            continue

                            total_contribution = contribution_1 + contribution_2
                            psi_2 += total_contribution

                            # Store contribution info
                            contributions.append({
                                'state': (n_prime, l_prime, m_prime),
                                'magnitude': np.abs(np.sum(total_contribution * dV))
                            })

                    except Exception as e:
                        print(f"Exception for first state ({n_prime},{l_prime},{m_prime}): {str(e)}")
                        continue

        # Print largest contributions
        if contributions:
            print("\nLargest second-order contributions:")
            for contrib in sorted(contributions, key=lambda x: x['magnitude'], reverse=True)[:5]:
                print(f"State {contrib['state']}: {contrib['magnitude']}")

        # Add all corrections and normalize
        psi_total = psi_0 + psi_2

        r_negative = r < 0
        theta_negative = theta < 0
        psi_total[r_negative] = np.conjugate(psi_total[r_negative])
        psi_total[theta_negative] = np.conjugate(psi_total[theta_negative])

        # Dann folgt die normale Normierung:
        norm = np.sqrt(np.sum(np.abs(psi_total) ** 2 * dV))
        if norm > 0:
            psi_total = psi_total / norm

        norm = np.sqrt(np.sum(np.abs(psi_total) ** 2 * dV))
        if norm > 0:
            psi_total = psi_total / norm
            print(f"\nFinal normalization factor: {norm}")

        return psi_total

    def total_correction_up_to_second_order(self, n, l, m, B, r, theta, phi):
        """
        Berechnet die Wellenfunktion mit Korrekturen bis zur zweiten Ordnung
        """
        psi_0 = self.wave_func(n, l, m, 1, r, theta, phi)
        psi_1 = self.first_order_correction(n, l, m, B, r, theta, phi) - psi_0  # Nur die Korrektur
        psi_2 = self.second_order_correction(n, l, m, B, r, theta, phi) - psi_0  # Nur die Korrektur

        # Kombiniere alle Beiträge
        psi_total = psi_0 + psi_1 + psi_2

        # Normierung
        dV = r ** 2 * np.sin(theta)
        norm = np.sqrt(np.sum(np.abs(psi_total) ** 2 * dV))
        if norm > 0:
            psi_total = psi_total / norm

        return psi_total

    def calculate_matrix_element(self, psi_k, psi_0, B, r, theta, phi, m):
        """Verbesserte Berechnung der Matrixelemente"""
        # Zeeman Term mit verbesserter m-Abhängigkeit
        zeeman = self.mu_B * B * m * np.conjugate(psi_k) * psi_0

        # Diamagnetischer Term mit präziserer Implementierung
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        r_perp_squared = x ** 2 + y ** 2

        # Verbesserte Behandlung des diamagnetischen Terms
        dia = (self.e ** 2 * B ** 2 / (8 * self.m_e)) * (
                np.conjugate(psi_k) * r_perp_squared * psi_0
        )

        return zeeman + dia

    def first_order_correction(self, n, l, m, B, r, theta, phi):
        """Verbesserte Version mit Gültigkeitsprüfung"""
        # Konvertierung in atomare Einheiten
        a0 = 5.29177e-11
        B_atomic = B * (a0 ** 2 * self.e / self.hbar)

        # Prüfe Gültigkeit der Störungstheorie
        characteristic_energy = self.calculate_orbitals_energy(n, 1)
        perturbation_strength = abs(self.mu_B * B_atomic)

        if perturbation_strength > 0.1 * abs(characteristic_energy):
            raise ValueError(f"""
            Störungstheorie nicht valide für dieses Magnetfeld!
            Störungsparameter: {perturbation_strength:.2e}
            Charakteristische Energie: {characteristic_energy:.2e}
            Bitte verwenden Sie B < {0.1 * abs(characteristic_energy) / self.mu_B:.2e} T
            """)

        # Rest der Implementierung wie zuvor...
        psi_0 = self.wave_func(n, l, m, 1, r, theta, phi)
        E_n = self.calculate_orbitals_energy(n, 1)

        # Initialisiere Korrektur
        psi_1 = np.zeros_like(psi_0, dtype=complex)
        dV = r ** 2 * np.sin(theta)

        # Tracking der Beiträge
        contributions = []

        # Summation über Zwischenzustände mit Konvergenzkontrolle
        max_contribution = 0
        total_contribution = 0

        cycle_contributions = 0

        for n_prime in range(max(1, n - 2), n + 3):
            for l_prime in range(max(0, l - 1), min(n_prime, l + 2)):
                for m_prime in range(-l_prime, l_prime + 1):
                    if (n_prime, l_prime, m_prime) == (n, l, m):
                        continue

                    try:
                        psi_k = self.wave_func(n_prime, l_prime, m_prime, 1, r, theta, phi)
                        E_k = self.calculate_orbitals_energy(n_prime, 1)

                        H_k0 = self.calculate_matrix_element(psi_k, psi_0, B_atomic, r, theta, phi, m)
                        integral = np.sum(H_k0 * dV)

                        delta_E = E_k - E_n
                        if abs(delta_E) > 1e-10:
                            contribution_magnitude = abs(integral / delta_E)

                            # Prüfe einzelne Beiträge
                            if contribution_magnitude > 0.1:
                                print(f"Warnung: Großer Beitrag von Zustand ({n_prime},{l_prime},{m_prime})")


                            contribution = (integral / delta_E) * psi_k
                            psi_1 += contribution

                            max_contribution = max(max_contribution, contribution_magnitude)
                            total_contribution += contribution_magnitude

                    except Exception as e:
                        continue

        # Finale Normierung mit Stabilisierung
        psi_total = psi_0 + psi_1
        norm = np.sqrt(np.sum(np.abs(psi_total) ** 2 * dV) + 1e-15)

        return psi_total / norm


    def check_normalization(self, n, l, r, Z, m, theta, phi, field):
        """
        Überprüft die Normierung der Wellenfunktion durch Integration
        der Wahrscheinlichkeitsdichte über den gesamten Raum.
        """
        # Berechne Wellenfunktion
        psi = self.first_order_approximation(n, l, r, Z, m, theta, phi, field)

        # Wahrscheinlichkeitsdichte
        density = np.abs(psi) ** 2

        # Volumenelement in Kugelkoordinaten
        dV = r ** 2 * np.sin(theta)

        # Berechne Integral über gesamten Raum
        total_prob = np.sum(density * dV)

        print(f"Gesamtwahrscheinlichkeit: {total_prob}")
        print(f"Abweichung von 1: {abs(1 - total_prob)}")

        return total_prob

    def wave_func_magneticfield(self, n, l, r, Z, m, theta, phi, field, approx):
        if approx:
            print("approx")
            # Unnormierte Version (alte Implementierung)
            psi_old = self.first_order_apporoximation_old(n, l, r, Z, m, theta, phi, field)
            density_old = np.abs(psi_old) ** 2

            # Normierte Version (neue Implementierung)
            psi_new = self.total_correction_up_to_second_order(n, l, r, Z, m, theta, phi, field)
            density_new = np.abs(psi_new) ** 2

            # Volumenelement
            dV = r ** 2 * np.sin(theta)

            # Integrale berechnen
            total_old = np.sum(density_old * dV)
            total_new = np.sum(density_new * dV)

            print("Vergleich der Normierung:")
            print(f"Alte Version - Gesamtwahrscheinlichkeit: {total_old}")
            print(f"Neue Version - Gesamtwahrscheinlichkeit: {total_new}")

            # Maximale Werte
            max_old = np.max(density_old)
            max_new = np.max(density_new)

            print(f"\nMaximale Dichte:")
            print(f"Alte Version: {max_old}")
            print(f"Neue Version: {max_new}")

            return psi_new

        else:
            print("no approx")
            psi_1 = self.first_order_correction(n, l, m, field, r, theta, phi)
            return psi_1

    def probability_density_magnetic_field(self, n, l, r, Z, m, theta, phi, field):
        """
        Improved probability density calculation for magnetic field effects
        """
        # Calculate wavefunction with perturbation
        psi = self.wave_func_magneticfield(n, l, r, Z, m, theta, phi, field,
                                           approx=self.visual_dict["Störtheorie Näherung"])

        # Calculate density with proper normalization
        density = np.abs(psi) ** 2

        # Volume element for proper normalization
        dV = r ** 2 * np.sin(theta)
        norm = np.sum(density * dV)

        if norm > 1e-10:  # Avoid division by very small numbers
            density = density / norm

        # Add small offset to prevent visualization artifacts
        density = density + np.max(density) * 0.001

        return density

    def validate_implementation(self):
        """Validierungstests für die Störtheorie"""

        # Test 1: Normierung
        def check_normalization(psi, r, theta, phi):
            dV = r ** 2 * np.sin(theta)
            norm = np.sum(np.abs(psi) ** 2 * dV)
            return abs(1 - norm)

        # Test 2: Symmetrie
        def check_symmetry(psi, m):
            # Für m=0 sollte die Wellenfunktion rotationssymmetrisch sein
            if m == 0:
                return np.allclose(psi[::2], psi[1::2], rtol=1e-5)
            return True

        # Test 3: Energieverschiebung
        def check_energy_shift(E_0, E_1, B):
            # Für kleine B sollte die Energieverschiebung linear mit B sein
            return abs(E_1 - E_0) < abs(B) * 1e-3

        # Führe Tests durch
        n, l, m = 1, 0, 0
        r = np.linspace(0, 10, 100)
        theta = np.linspace(0, np.pi, 100)
        phi = np.linspace(0, 2 * np.pi, 100)
        B = 0.1  # Schwaches Magnetfeld

        # Berechne Wellenfunktionen
        psi_0 = self.wave_func(n, l, m, 1, r, theta, phi)
        psi_1 = self.first_order_correction(n, l, m, B, r, theta, phi)

        # Führe Tests durch
        norm_error = check_normalization(psi_1, r, theta, phi)
        sym_check = check_symmetry(psi_1, m)

        return {
            'normalization_error': norm_error,
            'symmetry_check': sym_check,
        }

