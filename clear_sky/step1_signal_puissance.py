"""
Clear Sky - Détecteur de RFI par Spectral Kurtosis (SK)
ÉTAPE 1 : bruit gaussien complexe, injection de RFI, puissance.

Modèle : un seul canal fréquentiel, échantillons complexes x[n] (I + jQ).
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

M = 1024  # nombre d'échantillons par fenêtre d'analyse


# --------------------------------------------------------------------------
# 1) Bruit gaussien complexe stationnaire
# --------------------------------------------------------------------------
def bruit_gaussien_complexe(n, sigma2=1.0, rng=None):
    """x = I + jQ avec I, Q ~ N(0, sigma2/2) indépendants.

    Choix de sigma2/2 par composante : E|x|^2 = E[I^2] + E[Q^2] = sigma2.
    Donc sigma2 est directement la puissance moyenne du bruit.
    """
    rng = np.random.default_rng() if rng is None else rng
    i = rng.normal(0.0, np.sqrt(sigma2 / 2), n)
    q = rng.normal(0.0, np.sqrt(sigma2 / 2), n)
    return i + 1j * q


# --------------------------------------------------------------------------
# 2) Injection de RFI
# --------------------------------------------------------------------------
def inr_vers_amplitude(inr_db, sigma2=1.0):
    """INR = P_rfi / sigma2 (rapport interférence/bruit, en dB).
    Pour un phaseur d'amplitude A, P_rfi = A^2  =>  A = sqrt(INR_lin * sigma2).
    """
    return np.sqrt(10 ** (inr_db / 10) * sigma2)


def ajoute_rfi_impulsionnelle(x, debut, largeur, inr_db, sigma2=1.0, rng=None):
    """Salve courte : un phaseur constant, de phase aléatoire, actif sur
    `largeur` échantillons à partir de `debut`. INR défini PENDANT la salve."""
    rng = np.random.default_rng() if rng is None else rng
    y = x.copy()
    a = inr_vers_amplitude(inr_db, sigma2)
    phase = rng.uniform(0, 2 * np.pi)
    y[debut:debut + largeur] += a * np.exp(1j * phase)
    return y


def ajoute_rfi_sinusoidale(x, f_norm, inr_db, sigma2=1.0, rng=None):
    """Onde continue (CW) : a*exp(j(2*pi*f_norm*n + phi)), présente sur
    toute la fenêtre. f_norm = fréquence / fréquence d'échantillonnage."""
    rng = np.random.default_rng() if rng is None else rng
    n = np.arange(len(x))
    a = inr_vers_amplitude(inr_db, sigma2)
    phi = rng.uniform(0, 2 * np.pi)
    return x + a * np.exp(1j * (2 * np.pi * f_norm * n + phi))


# --------------------------------------------------------------------------
# 3) Puissance
# --------------------------------------------------------------------------
def puissance_instantanee(x):
    """P[n] = |x[n]|^2 = I^2 + Q^2."""
    return np.abs(x) ** 2


# --------------------------------------------------------------------------
# Démonstration + figure
# --------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    sigma2 = 1.0

    # --- Vérification statistique du bruit seul (beaucoup d'échantillons)
    x_long = bruit_gaussien_complexe(1_000_000, sigma2, rng)
    p_long = puissance_instantanee(x_long)
    print("Bruit seul, 1e6 échantillons :")
    print(f"  moyenne de P  = {p_long.mean():.4f}   (théorie : {sigma2})")
    print(f"  variance de P = {p_long.var():.4f}   (théorie : {sigma2**2})")

    # --- Une fenêtre de M échantillons : propre, impulsion, sinusoïde
    inr_db = 10.0
    x0 = bruit_gaussien_complexe(M, sigma2, rng)
    x_imp = ajoute_rfi_impulsionnelle(x0, debut=500, largeur=8,
                                      inr_db=inr_db, sigma2=sigma2, rng=rng)
    x_sin = ajoute_rfi_sinusoidale(x0, f_norm=0.1, inr_db=inr_db,
                                   sigma2=sigma2, rng=rng)

    cas = {"Sans RFI": x0,
           f"Impulsion (8 éch., INR={inr_db:.0f} dB)": x_imp,
           f"Sinusoïde (INR={inr_db:.0f} dB)": x_sin}
    print(f"\nFenêtre de M={M} échantillons, puissance moyenne :")
    for nom, x in cas.items():
        print(f"  {nom:35s} -> {puissance_instantanee(x).mean():.3f}")

    fig, ax = plt.subplots(2, 3, figsize=(14, 7))
    for k, (nom, x) in enumerate(cas.items()):
        p = puissance_instantanee(x)
        ax[0, k].plot(p, lw=0.7)
        ax[0, k].set_title(nom)
        ax[0, k].set_xlabel("échantillon n")
        ax[0, k].set_ylabel("P[n] = |x|²")
        ax[0, k].set_ylim(0, 25)
    # Histogramme de la puissance du bruit seul vs loi exponentielle
    bins = np.linspace(0, 8, 80)
    ax[1, 0].hist(p_long, bins=bins, density=True, alpha=0.6,
                  label="simulation (1e6)")
    pp = np.linspace(0, 8, 400)
    ax[1, 0].plot(pp, np.exp(-pp / sigma2) / sigma2, "r",
                  label=r"théorie $e^{-p/\sigma^2}/\sigma^2$")
    ax[1, 0].set_title("Loi de la puissance du bruit")
    ax[1, 0].set_xlabel("P")
    ax[1, 0].legend()
    # Histogrammes comparés des 3 cas
    for nom, x in cas.items():
        ax[1, 1].hist(puissance_instantanee(x), bins=bins, density=True,
                      histtype="step", label=nom)
    ax[1, 1].set_yscale("log")
    ax[1, 1].set_title("Histogramme de P : effet de la RFI")
    ax[1, 1].set_xlabel("P")
    ax[1, 1].legend(fontsize=7)
    # Spectre (pour voir la sinusoïde, utile pour l'intuition)
    for nom, x in cas.items():
        s = np.abs(np.fft.fftshift(np.fft.fft(x))) ** 2 / M
        ax[1, 2].plot(np.fft.fftshift(np.fft.fftfreq(M)), s, lw=0.6, label=nom)
    ax[1, 2].set_yscale("log")
    ax[1, 2].set_title("Spectre (FFT) - vue complémentaire")
    ax[1, 2].set_xlabel("fréquence normalisée")
    ax[1, 2].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig("/home/claude/clear_sky/step1_signal_puissance.png", dpi=110)
    print("\nFigure : /home/claude/clear_sky/step1_signal_puissance.png")
