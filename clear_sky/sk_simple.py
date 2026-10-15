"""
Clear Sky - Détecteur Spectral Kurtosis (SK), version minimale.

Un seul canal, bruit blanc gaussien complexe, une RFI sinusoïdale pendant un
intervalle connu, SK calculée sur des fenêtres successives de M = 1024 échantillons.
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(1)

M = 1024          # taille d'une fenêtre
N_FEN = 120       # nombre de fenêtres
N = M * N_FEN     # nombre total d'échantillons
SIGMA2 = 1.0      # puissance du bruit
INR_DB = 3.0      # rapport interférence/bruit
DEBUT, FIN = 60_500, 66_000   # intervalle de la RFI (en échantillons)

# 1) Bruit blanc gaussien complexe : x = I + jQ, E|x|^2 = SIGMA2
x = (rng.normal(0, np.sqrt(SIGMA2 / 2), N)
     + 1j * rng.normal(0, np.sqrt(SIGMA2 / 2), N))

# 2) RFI : onde sinusoïdale (fréquence normalisée 0.1) pendant [DEBUT, FIN)
amp = np.sqrt(10 ** (INR_DB / 10) * SIGMA2)
n = np.arange(DEBUT, FIN)
x[DEBUT:FIN] += amp * np.exp(1j * (2 * np.pi * 0.1 * n + rng.uniform(0, 2 * np.pi)))

# 3) Puissance instantanée, puis SK sur chaque fenêtre de M échantillons
P = np.abs(x) ** 2


def spectral_kurtosis(p):
    """SK = (M+1)/(M-1) * (M*S2/S1^2 - 1), S1 = somme(P), S2 = somme(P^2)."""
    m = len(p)
    s1, s2 = p.sum(), (p ** 2).sum()
    return (m + 1) / (m - 1) * (m * s2 / s1 ** 2 - 1)


SK = np.array([spectral_kurtosis(P[k * M:(k + 1) * M]) for k in range(N_FEN)])

# 4) Seuils à 3 sigma autour de 1 : sigma_SK ~ 2/sqrt(M)  (grand M, bruit gaussien)
sigma_sk = 2 / np.sqrt(M)
seuil_bas, seuil_haut = 1 - 3 * sigma_sk, 1 + 3 * sigma_sk
detecte = (SK < seuil_bas) | (SK > seuil_haut)

# Fenêtres contenant (au moins en partie) de la RFI : vérité terrain
fen_rfi = np.array([(k * M < FIN) and ((k + 1) * M > DEBUT) for k in range(N_FEN)])
print(f"Seuils : [{seuil_bas:.3f} ; {seuil_haut:.3f}]  (sigma_SK = {sigma_sk:.4f})")
print(f"Fenêtres avec RFI : {fen_rfi.sum()}  -> détectées : {(detecte & fen_rfi).sum()}")
print(f"Fenêtres sans RFI : {(~fen_rfi).sum()}  -> fausses alarmes : {(detecte & ~fen_rfi).sum()}")

# 5) Deux graphiques
t_fen = (np.arange(N_FEN) + 0.5) * M   # centre de chaque fenêtre
fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

ax[0].plot(P, lw=0.3, color="tab:blue", rasterized=True)
ax[0].axvspan(DEBUT, FIN, color="tab:red", alpha=0.15, label="intervalle RFI (connu)")
ax[0].set_ylabel("P[n] = |x[n]|²")
ax[0].set_title("Signal reçu : puissance en fonction du temps")
ax[0].legend(loc="upper right")

ax[1].plot(t_fen, SK, "o-", ms=3, lw=0.8, color="tab:blue", label="SK (M = 1024)")
ax[1].axhline(1, color="gray", ls=":", lw=1)
ax[1].axhline(seuil_haut, color="k", ls="--", lw=1, label=f"1 ± 3σ  ({seuil_bas:.3f} ; {seuil_haut:.3f})")
ax[1].axhline(seuil_bas, color="k", ls="--", lw=1)
ax[1].axvspan(DEBUT, FIN, color="tab:red", alpha=0.15)
ax[1].plot(t_fen[detecte], SK[detecte], "o", ms=7, mfc="none", mec="red", mew=1.5,
           label="fenêtre détectée (RFI)")
ax[1].set_xlabel("échantillon n")
ax[1].set_ylabel("SK")
ax[1].set_title("Spectral Kurtosis par fenêtre et seuils de détection")
ax[1].legend(loc="lower left")

plt.tight_layout()
plt.savefig("/home/claude/separation-algorithms-/clear_sky/sk_simple.png", dpi=110)
