"""
Clear Sky - Spectral Kurtosis dans un spectromètre FFT (NumPy + Matplotlib).

Principe du spectromètre :
  1. on découpe le signal en M blocs de NFFT échantillons ;
  2. chaque bloc est transformé par FFT -> un "spectre" de NFFT canaux ;
  3. pour chaque canal k, on a donc M valeurs de puissance P_m[k] = |X_m[k]|^2 ;
  4. on accumule : S1[k] = somme_m P_m[k]   et   S2[k] = somme_m P_m[k]^2 ;
  5. SK[k] = (M+1)/(M-1) * (M*S2[k]/S1[k]^2 - 1).

Rappel : pour un canal de bruit gaussien, P suit une loi exponentielle
(E[P^2] = 2 E[P]^2) donc SK = 1. Si P est plus "constante" que ça (sinusoïde),
SK < 1. Si P est plus "bouffie" (impulsions), SK > 1.
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================ PARAMÈTRES À MODIFIER ============================
NFFT = 64            # taille de la FFT = nombre de canaux fréquentiels
M = 1024             # nombre de spectres accumulés (taille de la fenêtre SK)
SIGMA2 = 1.0         # puissance du bruit par canal (échelle arbitraire)
K_GAUSS = 16         # canal contenant la composante gaussienne forte
K_CW = 40            # canal contenant la sinusoïde continue
GAUSS_DB = 20.0      # puissance de la composante gaussienne / bruit, dans son canal
CW_DB = 10.0         # puissance de la sinusoïde / bruit, dans son canal
CW_OFFSET = 0.0      # décalage de la sinusoïde par rapport au centre du canal
                     # (en largeur de canal, ex. 0.3). 0 = pile au centre.
SEED = 0
# ===============================================================================


def bruit_complexe(n, sigma2, rng):
    """Bruit blanc gaussien complexe, de puissance moyenne sigma2."""
    return (rng.normal(0, np.sqrt(sigma2 / 2), n)
            + 1j * rng.normal(0, np.sqrt(sigma2 / 2), n))


def composante_gaussienne(m, nfft, k, gauss_db, sigma2, rng):
    """Bruit gaussien à bande étroite, uniquement dans le canal k.

    Pour que toute l'énergie tombe dans UN canal, on prend une exponentielle
    complexe à la fréquence k/NFFT, dont l'amplitude complexe c est tirée au
    hasard (gaussienne) et change à chaque bloc : c'est un bruit gaussien dont
    le temps de cohérence vaut un bloc FFT.
    Normalisation : après FFT (divisée par sqrt(NFFT)), la puissance dans le
    canal vaut NFFT*var(c) -> on impose var(c) = rapport * sigma2 / NFFT.
    """
    rapport = 10 ** (gauss_db / 10)
    c = bruit_complexe(m, rapport * sigma2 / nfft, rng)          # un c par bloc
    porteuse = np.exp(2j * np.pi * k * np.arange(nfft) / nfft)  # fréquence k/NFFT
    return (c[:, None] * porteuse[None, :]).ravel()


def sinusoide_continue(m, nfft, k, offset, cw_db, sigma2, rng):
    """Sinusoïde d'amplitude constante, continue d'un bloc à l'autre.
    Amplitude A telle que la puissance dans le canal = rapport * sigma2."""
    rapport = 10 ** (cw_db / 10)
    a = np.sqrt(rapport * sigma2 / nfft)
    n = np.arange(m * nfft)                       # indice temporel global
    return a * np.exp(1j * (2 * np.pi * (k + offset) * n / nfft
                            + rng.uniform(0, 2 * np.pi)))


def spectrometre(x, nfft):
    """Découpe en blocs de NFFT, FFT de chaque bloc, puissance |X|^2.
    Retourne un tableau P de forme (M, NFFT) : P[m, k] = puissance du canal k
    dans le spectre numéro m. Division par sqrt(NFFT) : le bruit blanc de
    puissance sigma2 donne alors E[P] = sigma2 dans chaque canal."""
    blocs = x.reshape(-1, nfft)
    X = np.fft.fft(blocs, axis=1) / np.sqrt(nfft)
    return np.abs(X) ** 2


def accumule(P):
    """S1 = somme des puissances, S2 = somme des puissances au carré (sur les M spectres)."""
    return P.sum(axis=0), (P ** 2).sum(axis=0)


def sk(s1, s2, m):
    """Spectral Kurtosis standard."""
    return (m + 1) / (m - 1) * (m * s2 / s1 ** 2 - 1)


def simule():
    rng = np.random.default_rng(SEED)
    n = M * NFFT
    x = bruit_complexe(n, SIGMA2, rng)                                       # 1) bruit
    x = x + composante_gaussienne(M, NFFT, K_GAUSS, GAUSS_DB, SIGMA2, rng)   # 2) gaussien fort
    x = x + sinusoide_continue(M, NFFT, K_CW, CW_OFFSET, CW_DB, SIGMA2, rng) # 3) sinusoïde
    P = spectrometre(x, NFFT)                                                # 4) FFT + puissance
    s1, s2 = accumule(P)                                                     # 5) S1, S2
    return P, s1, s2, sk(s1, s2, M)                                          # 6) SK


if __name__ == "__main__":
    P, S1, S2, SK = simule()
    sigma_sk = 2 / np.sqrt(M)               # écart-type de SK pour du bruit gaussien
    canaux = np.arange(NFFT)
    bruit = np.ones(NFFT, bool)
    bruit[[K_GAUSS, K_CW]] = False

    # ------------------------------ Vérifications ------------------------------
    k_eff = S1[K_CW] / M / SIGMA2 - 1       # rapport sinusoïde/bruit mesuré
    sk_cw_theorie = (2 * k_eff + 1) / (1 + k_eff) ** 2
    print(f"Canaux de bruit seul  : SK moyenne = {SK[bruit].mean():.3f} "
          f"(attendu 1), écart-type = {SK[bruit].std():.3f} (attendu {sigma_sk:.3f})")
    print(f"Canal gaussien fort   : SK = {SK[K_GAUSS]:.3f} (attendu ~1 malgré {GAUSS_DB:.0f} dB)")
    print(f"Canal sinusoïde       : SK = {SK[K_CW]:.3f} (théorie ~{sk_cw_theorie:.3f}, "
          f"rapport mesuré K = {k_eff:.1f})")
    ok1 = abs(SK[K_GAUSS] - 1) < 3 * sigma_sk
    ok2 = SK[K_CW] < 1 - 3 * sigma_sk
    print(f"SK gaussien dans 1 ± 3σ : {ok1}   |   SK sinusoïde sous 1 - 3σ : {ok2}")

    # --------------------------------- Graphiques ---------------------------------
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    def marque(a):
        a.axvline(K_GAUSS, color="tab:orange", ls=":", lw=1)
        a.axvline(K_CW, color="tab:red", ls=":", lw=1)
        a.set_xlabel("canal k")

    ax[0, 0].semilogy(canaux, S1, "o-", ms=3)
    ax[0, 0].set_title("S1 : puissance accumulée")
    ax[0, 0].set_ylabel("S1[k] = Σ P")
    marque(ax[0, 0])

    ax[0, 1].semilogy(canaux, S2, "o-", ms=3, color="tab:green")
    ax[0, 1].set_title("S2 : puissances au carré accumulées")
    ax[0, 1].set_ylabel("S2[k] = Σ P²")
    marque(ax[0, 1])

    ax[1, 0].plot(canaux, SK, "o-", ms=3, color="tab:purple")
    ax[1, 0].axhline(1, color="gray", ls=":")
    ax[1, 0].axhline(1 + 3 * sigma_sk, color="k", ls="--", lw=1, label="1 ± 3σ")
    ax[1, 0].axhline(1 - 3 * sigma_sk, color="k", ls="--", lw=1)
    ax[1, 0].set_title("Spectral Kurtosis par canal")
    ax[1, 0].set_ylabel("SK[k]")
    ax[1, 0].legend()
    marque(ax[1, 0])
    ax[1, 0].set_ylim(0.1, 1.35)
    ax[1, 0].annotate("gaussien fort", (K_GAUSS, SK[K_GAUSS]), (K_GAUSS + 3, 0.6),
                      arrowprops=dict(arrowstyle="->"), color="tab:orange")
    ax[1, 0].annotate("sinusoïde", (K_CW, SK[K_CW]), (K_CW + 2, 0.55),
                      arrowprops=dict(arrowstyle="->"), color="tab:red")

    m_axe = np.arange(1, M + 1)
    for k, couleur, nom in [(K_GAUSS, "tab:orange", "canal gaussien fort"),
                            (K_CW, "tab:red", "canal sinusoïde")]:
        ax[1, 1].semilogy(m_axe, P[:, k], lw=0.4, color=couleur, alpha=0.35)
        ax[1, 1].semilogy(m_axe, np.cumsum(P[:, k]) / m_axe, lw=2, color=couleur,
                          label=f"{nom} : moyenne cumulée")
    ax[1, 1].set_title("Puissance au cours des accumulations")
    ax[1, 1].set_xlabel("nombre de spectres accumulés m")
    ax[1, 1].set_ylabel("P (fin) / moyenne cumulée (épais)")
    ax[1, 1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("/home/claude/separation-algorithms-/clear_sky/sk_fft_spectrometer.png", dpi=110)
