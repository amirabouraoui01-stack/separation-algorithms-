"""
Clear Sky - Sursaut solaire (modèle pédagogique) vs RFI OFDM (type LTE) vue par la SK.

On réutilise le spectromètre FFT de sk_fft_spectrometer.py :
bruit_complexe, spectrometre (FFT par blocs -> puissance), accumule (S1, S2), sk.

Trois scénarios, analysés SÉPARÉMENT (la superposition est dans sk_superposition.py) :
  1. "bruit"   : bruit gaussien complexe seul (référence statistique) ;
  2. "sursaut" : bruit récepteur + émission gaussienne dont la puissance monte, passe
                 par un maximum, puis décroît. MODÈLE PÉDAGOGIQUE : ce n'est pas une
                 simulation physique de sursaut solaire (pas de dérive en fréquence,
                 pas de structure spectrale, échelles de temps compressées) ;
  3. "ofdm"    : RFI OFDM simplifiée type LTE 20 MHz (15 kHz entre sous-porteuses,
                 1200 sous-porteuses QPSK, préfixe cyclique normal, trames de 1 ms).
                 Simplifications : pas de signaux de référence, pas de filtrage de mise
                 en forme, sous-porteuses toutes occupées (charge pleine).

Unités : puissances en unités arbitraires (u.a.), normalisées pour que le bruit vaille
SIGMA2 = 1 u.a. par canal (0 dB = niveau du bruit). FS, NFFT et les durées sont des
valeurs d'illustration inspirées de LTE, PAS les paramètres d'ORFEES : ce qui compte
pour la SK est le rapport (durée des phénomènes) / (durée d'accumulation M*DT).
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from sk_fft_spectrometer import bruit_complexe, spectrometre, accumule, sk

# ============================ PARAMÈTRES À MODIFIER ============================
FS = 30.72e6          # fréquence d'échantillonnage (Hz) - valeur LTE 20 MHz, illustrative
NFFT = 64             # canaux du spectromètre -> largeur de canal FS/NFFT = 480 kHz
N_BLOCKS = 16384      # nombre total de spectres simulés (durée d'observation)
SIGMA2 = 1.0          # puissance du bruit (u.a.)
M_REF = 1024          # nombre d'accumulations de référence
M_LIST = (64, 256, 1024, 4096)   # valeurs de M testées

# Sursaut (profil temporel exprimé en indices de spectres)
BURST_START = 5000    # début de la montée
BURST_RISE = 200      # durée de la montée (spectres)
BURST_TAU = 1000      # constante de temps de la décroissance exponentielle (spectres)
BURST_PEAK_DB = 8.0   # puissance totale au maximum / bruit (dB)
BURST_ACTIVE_FRAC = 0.2   # "sursaut présent" si l'excès de puissance dépasse 20 % du pic

# OFDM type LTE 20 MHz
OFDM_DB = 5.0         # puissance du signal / bruit dans un canal occupé (dB)
OFDM_NFFT = 2048      # taille de l'IFFT OFDM (15 kHz = FS/2048)
OFDM_NUSED = 1200     # sous-porteuses utilisées (+-600, DC exclue)
OFDM_ON_PROB = 0.5    # variante "intermittente" : probabilité qu'une sous-trame (1 ms) soit active
SEED = 0
# ===============================================================================

DT = NFFT / FS                          # durée d'un spectre (s)
SUBFRAME = 30720                        # échantillons par sous-trame LTE (1 ms)
CP_SLOT = [160, 144, 144, 144, 144, 144, 144]   # préfixes cycliques normaux d'un slot
SEUIL_N_SIGMA = 3                       # seuils à 3 sigma, sigma_SK = 2/sqrt(M)
CAN_CENTRE = NFFT // 2                  # canal DC (centre de la bande OFDM)
CAN_HORS_BANDE = NFFT // 2 + 24         # canal hors de la bande OFDM


# ------------------------------------------------------------------ scénarios ---
def profil_sursaut(n_blocks, start, rise, tau):
    """Forme du sursaut f(m) dans [0,1] : montée lisse (sin^2), maximum, décroissance
    exponentielle. m = indice du spectre."""
    m = np.arange(n_blocks)
    f = np.zeros(n_blocks)
    up = (m >= start) & (m < start + rise)
    f[up] = np.sin(0.5 * np.pi * (m[up] - start) / rise) ** 2
    dec = m >= start + rise
    f[dec] = np.exp(-(m[dec] - start - rise) / tau)
    return f


def scenario_bruit(rng):
    """Référence : bruit blanc gaussien complexe stationnaire."""
    return bruit_complexe(N_BLOCKS * NFFT, SIGMA2, rng)


def scenario_sursaut(rng, peak_db=None):
    """Bruit récepteur (constant) + émission solaire gaussienne de puissance
    a*f(m)*SIGMA2 : somme de deux gaussiennes indépendantes -> gaussienne de variance
    SIGMA2*(1 + a*f(m)). La puissance varie lentement, la statistique reste gaussienne
    à chaque instant."""
    peak_db = BURST_PEAK_DB if peak_db is None else peak_db
    f = profil_sursaut(N_BLOCKS, BURST_START, BURST_RISE, BURST_TAU)
    a = 10 ** (peak_db / 10) - 1            # excès de puissance au pic
    n = N_BLOCKS * NFFT
    amplitude = np.sqrt(a * np.repeat(f, NFFT))      # une valeur par échantillon
    x = bruit_complexe(n, SIGMA2, rng) + amplitude * bruit_complexe(n, SIGMA2, rng)
    return x, f, a


def ofdm_lte(n_samples, rng, active=None, modulation="qpsk", n_fft=OFDM_NFFT,
             n_used=OFDM_NUSED, level_db=None):
    """Signal OFDM type LTE, échantillonné à FS = n_fft * 15 kHz.

    Chaque symbole OFDM : sous-porteuses modulées (QPSK ou 16-QAM, tirage aléatoire),
    IFFT, puis préfixe cyclique (copie de la fin du symbole placée devant).
    active : tableau booléen par sous-trame (1 ms) ; None = toujours actif.
    Normalisation : puissance dans un canal occupé du spectromètre = level_db / bruit.
    """
    level_db = OFDM_DB if level_db is None else level_db
    n_sf = int(np.ceil(n_samples / SUBFRAME))
    n_sym = n_sf * 14                                   # 14 symboles par sous-trame
    if modulation == "qpsk":
        niv = np.array([-1.0, 1.0]) / np.sqrt(2)
    else:                                               # 16-QAM : 4 niveaux par axe
        niv = np.array([-3.0, -1.0, 1.0, 3.0]) / np.sqrt(10)
    c = rng.choice(niv, (n_sym, n_used)) + 1j * rng.choice(niv, (n_sym, n_used))
    idx = np.r_[1:n_used // 2 + 1, -n_used // 2:0]      # sous-porteuses, DC exclue
    X = np.zeros((n_sym, n_fft), complex)
    X[:, idx % n_fft] = c
    s = np.fft.ifft(X, axis=1)                          # un symbole par ligne
    cp = [int(round(k * n_fft / OFDM_NFFT)) for k in CP_SLOT]
    sig = np.concatenate([np.concatenate([s[i, -cp[i % 7]:], s[i]]) for i in range(n_sym)])
    # puissance totale par échantillon = niveau_canal * SIGMA2 * (fraction de bande occupée)
    cible = 10 ** (level_db / 10) * SIGMA2 * n_used / n_fft
    sig *= np.sqrt(cible / np.mean(np.abs(sig) ** 2))
    if active is not None:
        sig *= np.repeat(active, SUBFRAME * n_fft // OFDM_NFFT)[:len(sig)]
    return sig[:n_samples]


def scenario_ofdm(rng, intermittent=False):
    """Bruit + RFI OFDM. intermittent=True : sous-trames actives/inactives au hasard."""
    n = N_BLOCKS * NFFT
    n_sf = int(np.ceil(n / SUBFRAME))
    active = (rng.random(n_sf) < OFDM_ON_PROB) if intermittent else None
    return bruit_complexe(n, SIGMA2, rng) + ofdm_lte(n, rng, active), active


# -------------------------------------------------------------------- analyse ---
def puissances(x, nfft=NFFT):
    """FFT successives -> P[m, k], canaux rangés de -FS/2 à +FS/2 (fftshift)."""
    return np.fft.fftshift(spectrometre(x, nfft), axes=1)


def sk_fenetres(P, m):
    """S1, S2 et SK par fenêtre de m spectres consécutifs : formes (n_fen, n_canaux)."""
    n_fen = P.shape[0] // m
    Pw = P[:n_fen * m].reshape(n_fen, m, -1)
    s1 = Pw.sum(axis=1)
    s2 = (Pw ** 2).sum(axis=1)
    return s1, s2, sk(s1, s2, m)


def seuils(m):
    """Seuils 1 +- 3 sigma, sigma_SK = 2/sqrt(m) (bruit gaussien, m grand)."""
    s = 2 / np.sqrt(m)
    return 1 - SEUIL_N_SIGMA * s, 1 + SEUIL_N_SIGMA * s


def sk_theorique(mu, m, extra=0.0):
    """SK attendue pour un canal gaussien dont la puissance moyenne mu[m] varie
    lentement (constante dans un spectre) : P_m = mu_m * E_m avec E_m exponentielle.
        E[P^2]/E[P]^2 = 2 <mu^2>/<mu>^2   ->   SK ~ 2<mu^2>/<mu>^2 - 1 = 1 + 2 CV^2.
    extra : puissance constante ajoutée (ex. une RFI gaussienne stationnaire)."""
    n_fen = len(mu) // m
    w = mu[:n_fen * m].reshape(n_fen, m) + extra
    return (m + 1) / (m - 1) * (2 * (w ** 2).mean(1) / w.mean(1) ** 2 - 1)


def masques(nom, f, m, n_fen):
    """Vérité terrain par (fenêtre, canal) : 'affecte' (le phénomène est présent) et
    'propre' (certainement sans phénomène). Les canaux de bord de bande OFDM (fuite
    spectrale, canal partiellement occupé) ne sont comptés dans aucun des deux."""
    tous = np.ones((n_fen, NFFT), bool)
    d = np.abs(np.arange(NFFT) - NFFT // 2)
    if nom == "bruit":
        return ~tous, tous
    if nom == "sursaut":
        act = f[:n_fen * m].reshape(n_fen, m).max(axis=1) > BURST_ACTIVE_FRAC
        return act[:, None] & tous, ~act[:, None] & tous
    # OFDM : bande occupée = |canal| <= 18 ; hors bande = |canal| >= 21
    return (tous & (d <= 18)), (tous & (d >= 21))


def genere(nom, rng):
    """Retourne P (spectres décalés), profil f (sursaut) ou activité (OFDM)."""
    if nom == "bruit":
        return puissances(scenario_bruit(rng)), None
    if nom == "sursaut":
        x, f, _ = scenario_sursaut(rng)
        return puissances(x), f
    x, active = scenario_ofdm(rng, intermittent=(nom == "ofdm_inter"))
    return puissances(x), active


def etude_duree_accumulation(noms, n_real=10, seed=SEED):
    """Taux de détection et de fausses alarmes (seuils 3 sigma) pour chaque M,
    moyennés sur n_real réalisations indépendantes."""
    cpt = {n: {m: [0, 0, 0, 0] for m in M_LIST} for n in noms}   # det, n_aff, fa, n_prop
    for r in range(n_real):
        rng = np.random.default_rng(seed + 1000 + r)
        for nom in noms:
            P, extra = genere(nom, rng)
            f = extra if nom == "sursaut" else None
            for m in M_LIST:
                _, _, SK = sk_fenetres(P, m)
                lo, hi = seuils(m)
                drap = (SK < lo) | (SK > hi)
                aff, pro = masques("sursaut" if nom == "sursaut" else
                                   ("bruit" if nom == "bruit" else "ofdm"), f, m, SK.shape[0])
                c = cpt[nom][m]
                c[0] += (drap & aff).sum(); c[1] += aff.sum()
                c[2] += (drap & pro).sum(); c[3] += pro.sum()
    return {n: {m: (c[0] / c[1] if c[1] else np.nan, c[2] / c[3] if c[3] else np.nan)
                for m, c in d.items()} for n, d in cpt.items()}


def sk_ofdm_resolution(n_fft_spectro, modulation, m=128, seed=SEED):
    """SK moyenne dans la bande OFDM selon la résolution spectrale du spectromètre :
    nombre de sous-porteuses par canal = 2048 / n_fft_spectro."""
    rng = np.random.default_rng(seed + 7)
    n = 2 ** 20
    x = bruit_complexe(n, SIGMA2, rng) + ofdm_lte(n, rng, None, modulation)
    P = puissances(x, n_fft_spectro)
    _, _, SK = sk_fenetres(P, m)
    d = np.abs(np.arange(n_fft_spectro) - n_fft_spectro // 2)
    return SK[:, d <= 0.27 * n_fft_spectro].mean(), SK[:, d <= 0.27 * n_fft_spectro].std()


# ------------------------------------------------------------------ graphiques ---
COUL = {"bruit": "tab:gray", "sursaut": "tab:orange", "ofdm": "tab:blue",
        "ofdm_inter": "tab:green"}
TITRE = {"bruit": "1. Bruit seul", "sursaut": "2. Sursaut (modèle pédagogique)",
         "ofdm": "3. RFI OFDM (type LTE, charge pleine)"}
F_MHZ = (np.arange(NFFT) - NFFT // 2) * FS / NFFT / 1e6      # fréquence des canaux (MHz)
T_FEN_MS = M_REF * DT * 1e3                                    # durée d'une fenêtre (ms)


def lissage(v, n):
    return np.convolve(v, np.ones(n) / n, mode="same")


def figure_ensemble(data, chemin):
    noms = ["bruit", "sursaut", "ofdm"]
    fig = plt.figure(figsize=(15, 19))
    gs = GridSpec(5, 3, figure=fig, height_ratios=[1.1, 1, 1, 1, 1.1], hspace=0.42, wspace=0.25)
    n_fen = N_BLOCKS // M_REF
    t_ms = np.arange(N_BLOCKS) * DT * 1e3
    lo, hi = seuils(M_REF)
    # fenêtre mise en évidence : celle où le sursaut modifie le plus la SK
    _, _, SK_s = sk_fenetres(data["sursaut"][0], M_REF)
    fen_pic = int(np.argmax(SK_s.mean(axis=1)))
    sk_all, aff_all = {}, {}
    for j, nom in enumerate(noms):
        P, extra = data[nom]
        _, _, SK = sk_fenetres(P, M_REF)
        sk_all[nom] = SK
        # --- (1) spectrogramme : moyenne par paquets de 32 spectres, en dB / bruit
        pq = 32
        Pm = P[:N_BLOCKS // pq * pq].reshape(-1, pq, NFFT).mean(axis=1)
        ax = fig.add_subplot(gs[0, j])
        im = ax.imshow(10 * np.log10(Pm.T / SIGMA2), origin="lower", aspect="auto", cmap="viridis",
                       extent=[0, N_BLOCKS * DT * 1e3, F_MHZ[0], F_MHZ[-1]], vmin=-1, vmax=7)
        for k in range(1, n_fen):
            ax.axvline(k * T_FEN_MS, color="w", lw=0.4, alpha=0.5)
        ax.set_title(TITRE[nom], fontsize=11)
        ax.set_xlabel("temps (ms)")
        ax.set_ylabel("fréquence relative (MHz)")
        if j == 2:
            cb = fig.colorbar(im, ax=ax, pad=0.02)
            cb.set_label("puissance (dB / bruit)")
        ax.text(0.01, 0.97, f"traits blancs : fenêtres de M={M_REF} spectres ({T_FEN_MS:.2f} ms)\n"
                f"moyenne par paquets de {pq} spectres", transform=ax.transAxes, color="w",
                fontsize=7, va="top")
        # --- (2) spectre de puissance moyenne
        ax = fig.add_subplot(gs[1, j])
        ax.plot(F_MHZ, 10 * np.log10(P.mean(axis=0) / SIGMA2), color=COUL[nom], lw=1.5)
        ax.axhline(0, color="k", ls=":", lw=0.8)
        ax.set_ylim(-1, 8)
        ax.set_xlabel("fréquence relative (MHz)")
        ax.set_ylabel("puissance moyenne (dB / bruit)")
        ax.set_title(f"spectre moyen S1/N sur {N_BLOCKS} spectres", fontsize=9)
        # --- (3) spectre de SK : toutes les fenêtres en gris, fenêtre du pic en couleur
        ax = fig.add_subplot(gs[2, j])
        ax.axhspan(lo, hi, color="k", alpha=0.08, label=f"1 ± 3σ (M={M_REF})")
        for w in range(n_fen):
            ax.plot(F_MHZ, SK[w], color="gray", lw=0.5, alpha=0.5)
        ax.plot(F_MHZ, SK[fen_pic], color=COUL[nom] if nom != "bruit" else "k", lw=1.8,
                label=f"fenêtre {fen_pic} ({fen_pic * T_FEN_MS:.1f}–{(fen_pic + 1) * T_FEN_MS:.1f} ms, montée du sursaut)")
        ax.axhline(1, color="k", ls=":", lw=0.8)
        ax.set_ylim(0.3, 2.7)
        ax.set_xlabel("fréquence relative (MHz)")
        ax.set_ylabel(f"SK (M = {M_REF})")
        ax.legend(fontsize=7, loc="lower left")
        ax.set_title("SK par canal (gris : les 16 fenêtres)", fontsize=9)
        # --- (4) évolution temporelle de la puissance
        ax = fig.add_subplot(gs[3, j])
        for can, st in [(CAN_CENTRE, "-"), (CAN_HORS_BANDE, "--")]:
            ax.plot(t_ms[64:-64:8], 10 * np.log10(lissage(P[:, can], 64)[64:-64:8] / SIGMA2), st,
                    color=COUL[nom] if nom != "bruit" else "k", lw=1.2,
                    label=f"canal {F_MHZ[can]:+.1f} MHz")
        for k in range(1, n_fen):
            ax.axvline(k * T_FEN_MS, color="gray", lw=0.4, alpha=0.5)
        ax.set_ylim(-2, 11)
        ax.set_xlabel("temps (ms)")
        ax.set_ylabel("puissance (dB / bruit)")
        ax.set_title("puissance dans 2 canaux (moyenne glissante 64 spectres)", fontsize=9)
        ax.legend(fontsize=7, loc="upper right", ncol=2)
        # sous-ensemble pour la distribution
        _, _, SKm = sk_fenetres(P, M_REF)
        aff, _ = masques("sursaut" if nom == "sursaut" else ("bruit" if nom == "bruit" else "ofdm"),
                         extra if nom == "sursaut" else None, M_REF, n_fen)
        aff_all[nom] = SKm[aff] if nom != "bruit" else SKm.ravel()
    # --- (5) distributions comparées
    ax = fig.add_subplot(gs[4, :])
    bins = np.linspace(0.5, 2.5, 61)
    lab = {"bruit": "bruit seul (tous canaux, toutes fenêtres)",
           "sursaut": "sursaut (fenêtres où le sursaut est présent, tous canaux)",
           "ofdm": "OFDM (canaux de la bande occupée, toutes fenêtres)"}
    for nom in noms:
        v = aff_all[nom]
        ax.hist(v, bins=bins, density=True, histtype="stepfilled", alpha=0.35, color=COUL[nom])
        ax.hist(v, bins=bins, density=True, histtype="step", color=COUL[nom], lw=1.5,
                label=f"{lab[nom]} : moyenne {v.mean():.2f}, écart-type {v.std():.2f}, N={len(v)}")
    ax.axvline(lo, color="k", ls="--"); ax.axvline(hi, color="k", ls="--", label=f"seuils 1 ± 3σ, M={M_REF}")
    ax.set_xlabel(f"SK (M = {M_REF} spectres)")
    ax.set_ylabel("densité")
    ax.set_title("Comparaison des distributions de SK", fontsize=10)
    ax.legend(fontsize=8)
    fig.suptitle(f"Clear Sky - SK : bruit, sursaut pédagogique, OFDM type LTE   "
                 f"(FS = {FS / 1e6:.2f} MHz, NFFT = {NFFT}, canal = {FS / NFFT / 1e3:.0f} kHz, "
                 f"1 spectre = {DT * 1e6:.2f} µs, bruit = 0 dB)", fontsize=11, y=0.905)
    fig.savefig(chemin, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return sk_all


def figure_duree(data, taux, chemin):
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    noms = ["bruit", "sursaut", "ofdm", "ofdm_inter"]
    leg = {"bruit": "bruit seul", "sursaut": "sursaut", "ofdm": "OFDM charge pleine",
           "ofdm_inter": f"OFDM intermittent (p={OFDM_ON_PROB})"}
    # (a) SK du canal central pour chaque fenêtre, en fonction de M
    for i, nom in enumerate(noms):
        P, _ = data[nom]
        for m in M_LIST:
            _, _, SK = sk_fenetres(P, m)
            x = m * (1 + 0.06 * (i - 1.5))
            ax[0].plot(np.full(SK.shape[0], x), SK[:, CAN_CENTRE], ".", color=COUL[nom], ms=4,
                       alpha=0.6, label=leg[nom] if m == M_LIST[0] else None)
    ms = np.array(M_LIST, float)
    mm = np.geomspace(M_LIST[0] * 0.8, M_LIST[-1] * 1.2, 100)
    ax[0].fill_between(mm, 1 - 3 * 2 / np.sqrt(mm), 1 + 3 * 2 / np.sqrt(mm), color="k", alpha=0.1,
                       label="1 ± 3σ (σ = 2/√M)")
    ax[0].set_xscale("log"); ax[0].set_xticks(ms); ax[0].set_xticklabels([str(int(v)) for v in ms])
    ax[0].set_ylim(0, 3.2)
    ax[0].set_xlabel("M (spectres accumulés)")
    ax[0].set_ylabel("SK du canal central, une valeur par fenêtre")
    ax[0].set_title("(a) SK vs durée d'accumulation", fontsize=10)
    ax[0].legend(fontsize=7, loc="upper left")
    sec = ax[0].secondary_xaxis("top", functions=(lambda v: v * DT * 1e3, lambda v: v / (DT * 1e3)))
    sec.set_xlabel("durée d'accumulation M·DT (ms)")
    # (b) taux de détection / fausses alarmes
    for nom in ["sursaut", "ofdm", "ofdm_inter", "bruit"]:
        det = [taux[nom][m][0] * 100 for m in M_LIST]
        fa = [taux[nom][m][1] * 100 for m in M_LIST]
        if nom != "bruit":
            ax[1].plot(M_LIST, det, "o-", color=COUL[nom], label=f"détection : {leg[nom]}")
        ax[1].plot(M_LIST, fa, "s--", color=COUL[nom], ms=4, alpha=0.8,
                   label=f"fausses alarmes : {leg[nom]}")
    ax[1].set_xscale("log"); ax[1].set_xticks(ms); ax[1].set_xticklabels([str(int(v)) for v in ms])
    ax[1].set_ylim(-3, 105)
    ax[1].set_xlabel("M (spectres accumulés)")
    ax[1].set_ylabel("taux (%)")
    ax[1].set_title("(b) détection / fausses alarmes, seuils 3σ, 10 réalisations", fontsize=10)
    ax[1].legend(fontsize=6.5, loc="center left")
    # (c) OFDM : SK selon la résolution spectrale
    nf = [16, 64, 256, 1024, 2048]
    for mod, c in [("qpsk", "tab:blue"), ("16qam", "tab:purple")]:
        v = [sk_ofdm_resolution(n, mod) for n in nf]
        ax[2].errorbar([2048 // n for n in nf], [a for a, _ in v], yerr=[b for _, b in v],
                       fmt="o-", color=c, capsize=3, label=mod.upper())
    ax[2].axhspan(1 - 3 * 2 / np.sqrt(128), 1 + 3 * 2 / np.sqrt(128), color="k", alpha=0.1,
                  label="1 ± 3σ (M=128)")
    ax[2].set_xscale("log", base=2)
    ax[2].set_xlabel("sous-porteuses OFDM par canal du spectromètre")
    ax[2].set_ylabel("SK moyenne dans la bande (M = 128)")
    ax[2].set_title("(c) OFDM : la SK dépend de la résolution spectrale", fontsize=10)
    ax[2].legend(fontsize=8)
    fig.suptitle(f"Influence de la durée d'accumulation (1 spectre = {DT * 1e6:.2f} µs ; "
                 f"1 sous-trame LTE = {SUBFRAME / FS * 1e3:.0f} ms = {SUBFRAME // NFFT} spectres)",
                 fontsize=11)
    plt.tight_layout()
    fig.savefig(chemin, dpi=100, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------------ main ---
if __name__ == "__main__":
    dossier = "/home/claude/separation-algorithms-/clear_sky/"
    rng = np.random.default_rng(SEED)
    data = {nom: genere(nom, rng) for nom in ["bruit", "sursaut", "ofdm", "ofdm_inter"]}
    n_fen = N_BLOCKS // M_REF
    lo, hi = seuils(M_REF)
    print(f"Seuils M={M_REF} : [{lo:.3f} ; {hi:.3f}]   1 spectre = {DT*1e6:.2f} µs, "
          f"fenêtre = {T_FEN_MS:.2f} ms, sous-trame LTE = {SUBFRAME // NFFT} spectres")

    sk_all = figure_ensemble(data, dossier + "sk_scenarios_ensemble.png")

    # --- vérifications chiffrées (M = M_REF) ---
    print("\n[Bruit] SK moyenne (tous canaux/fenêtres) = %.3f, écart-type = %.3f (théorie 1, %.3f)"
          % (sk_all["bruit"].mean(), sk_all["bruit"].std(), 2 / np.sqrt(M_REF)))
    P, f = data["sursaut"]
    a = 10 ** (BURST_PEAK_DB / 10) - 1
    th = sk_theorique(SIGMA2 * (1 + a * f), M_REF)
    print("[Sursaut] SK par fenêtre : mesure (moyenne des canaux) vs théorie 1+2CV²")
    for w in range(n_fen):
        print(f"   fen {w:2d} ({w*T_FEN_MS:5.1f} ms) : mesure {sk_all['sursaut'][w].mean():.3f} "
              f"± {sk_all['sursaut'][w].std():.3f}   théorie {th[w]:.3f}")
    SKo = sk_all["ofdm"]
    d = np.abs(np.arange(NFFT) - NFFT // 2)
    print("[OFDM charge pleine] SK dans la bande : moyenne %.3f, écart-type %.3f ; hors bande : "
          "moyenne %.3f" % (SKo[:, d <= 18].mean(), SKo[:, d <= 18].std(), SKo[:, d >= 21].mean()))
    _, _, SKi = sk_fenetres(data["ofdm_inter"][0], M_REF)
    print("[OFDM intermittent] SK dans la bande : moyenne %.3f, min %.3f, max %.3f"
          % (SKi[:, d <= 18].mean(), SKi[:, d <= 18].mean(axis=1).min(), SKi[:, d <= 18].mean(axis=1).max()))

    taux = etude_duree_accumulation(["bruit", "sursaut", "ofdm", "ofdm_inter"], n_real=10)
    print("\nTaux (%) par M : détection / fausses alarmes (10 réalisations, seuils 3σ)")
    print("  " + "".ljust(12) + "".join(f"M={m:<14}" for m in M_LIST))
    for nom in taux:
        print("  " + nom.ljust(12) + "".join(
            f"{taux[nom][m][0]*100:5.1f}/{taux[nom][m][1]*100:4.1f}    " for m in M_LIST))
    figure_duree(data, taux, dossier + "sk_scenarios_duree_M.png")
    print("\nFigures : sk_scenarios_ensemble.png, sk_scenarios_duree_M.png")
