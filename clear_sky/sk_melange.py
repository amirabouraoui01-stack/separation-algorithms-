"""
Clear Sky - Quatre scénarios sur les MÊMES réalisations :
    1. bruit            : x = n
    2. sursaut + bruit  : x = n + s
    3. OFDM + bruit     : x = n + r
    4. mélange          : x = n + s + r      <- nouveau

n = bruit récepteur, s = émission solaire gaussienne (profil de sursaut pédagogique),
r = RFI OFDM type LTE (charge pleine). Les trois composantes sont tirées UNE fois par
réalisation puis additionnées dans le domaine temporel AVANT la FFT, comme le ferait un
récepteur. Les différences entre scénarios ne viennent donc que de la présence ou non
d'une composante, pas du hasard du tirage.

La SK du mélange est TOUJOURS recalculée sur le signal total : la SK n'est pas additive
(elle dépend du rapport S2/S1^2, non linéaire en la puissance).

Les modèles (sursaut, OFDM), les paramètres (FS = 30,72 MHz, NFFT = 64, M = 1024) et
les fonctions viennent de sk_scenarios.py (lui-même basé sur sk_fft_spectrometer.py).

Unités : puissance en dB par rapport au bruit (0 dB = bruit), temps en ms,
fréquence relative au centre de bande en MHz.
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import sk_scenarios as sc
from sk_scenarios import (FS, NFFT, N_BLOCKS, SIGMA2, M_REF, DT, F_MHZ, T_FEN_MS, SEED,
                          BURST_START, BURST_RISE, BURST_TAU, BURST_PEAK_DB, BURST_ACTIVE_FRAC,
                          OFDM_DB, bruit_complexe, ofdm_lte, profil_sursaut, puissances,
                          sk_fenetres, seuils, sk_theorique, lissage)
from sk_fft_spectrometer import sk

# ============================ PARAMÈTRES À MODIFIER ============================
N_REAL = 20                         # réalisations indépendantes pour les taux
W_SHOW = [2, 4, 5, 6, 10]           # fenêtres tracées en SK(f) (calme, début, pic, fin, après)
CAN_BANDE = NFFT // 2 + 8           # canal étudié (dans la bande OFDM, hors canal DC)
NIVEAUX_OFDM = np.arange(-10, 21, 5)   # niveaux OFDM (dB / bruit) du balayage
N_REAL_BALAYAGE = 10
# ===============================================================================

NOMS = ["bruit", "sursaut", "ofdm", "melange"]
TITRES = {"bruit": "1. Bruit seul", "sursaut": "2. Sursaut + bruit",
          "ofdm": "3. OFDM + bruit", "melange": "4. Bruit + sursaut + OFDM"}
COUL = {"bruit": "tab:gray", "sursaut": "tab:orange", "ofdm": "tab:blue", "melange": "tab:red"}
A_SURSAUT_ON = {"sursaut", "melange"}
OFDM_ON = {"ofdm", "melange"}
D_CAN = np.abs(np.arange(NFFT) - NFFT // 2)
BANDE = D_CAN <= 18          # canaux entièrement dans la bande OFDM (+-600 sous-porteuses = +-9 MHz)
HORS = D_CAN >= 21           # canaux hors bande (on écarte 19-20 : bord de bande et fuite spectrale)
BANDE_MHZ = 600 * 15e3 / 1e6  # demi-largeur de la bande occupée (MHz)
ETAT_NOM = {0: "ni l'un ni l'autre", 1: "sursaut seul", 2: "OFDM seule", 3: "sursaut + OFDM"}


# ------------------------------------------------------------------- signaux ---
def composantes(rng):
    """Tire UNE FOIS bruit n, émission solaire s et RFI OFDM r (même réalisation pour
    les quatre scénarios). s = sqrt(a f(m)) * gaussienne : puissance a*f(m)*SIGMA2."""
    n = N_BLOCKS * NFFT
    f = profil_sursaut(N_BLOCKS, BURST_START, BURST_RISE, BURST_TAU)
    a = 10 ** (BURST_PEAK_DB / 10) - 1
    n_rx = bruit_complexe(n, SIGMA2, rng)
    sun = np.sqrt(a * np.repeat(f, NFFT)) * bruit_complexe(n, SIGMA2, rng)
    rfi = ofdm_lte(n, rng, None)                 # OFDM charge pleine, OFDM_DB dB / bruit
    return {"n": n_rx, "s": sun, "r": rfi, "f": f, "a": a}


def signal_total(c, nom, gain_rfi=1.0):
    """Somme temporelle des composantes présentes dans le scénario (avant la FFT)."""
    x = c["n"].copy()
    if nom in A_SURSAUT_ON:
        x = x + c["s"]
    if nom in OFDM_ON:
        x = x + gain_rfi * c["r"]
    return x


def periode_sursaut():
    """Début et fin (en indices de spectres) du sursaut : f > BURST_ACTIVE_FRAC."""
    return BURST_START, BURST_START + BURST_RISE + BURST_TAU * np.log(1 / BURST_ACTIVE_FRAC)


# --------------------------------------------------------------- vérité terrain ---
def etats(nom, f, m=M_REF):
    """Pour chaque cellule (fenêtre, canal) : 0 rien, 1 sursaut seul, 2 OFDM seule,
    3 les deux, -1 = canal de bord de bande (exclu des statistiques).
    Sursaut présent dans une fenêtre si f > BURST_ACTIVE_FRAC pour au moins un spectre ;
    OFDM présente dans les canaux de la bande occupée (elle est continue dans le temps)."""
    n_fen = N_BLOCKS // m
    act = f[:n_fen * m].reshape(n_fen, m).max(axis=1) > BURST_ACTIVE_FRAC
    e = np.zeros((n_fen, NFFT), int)
    if nom in A_SURSAUT_ON:
        e += act[:, None].astype(int)
    if nom in OFDM_ON:
        e += 2 * BANDE[None, :].astype(int)
    e[:, ~(BANDE | HORS)] = -1
    return e, act


def drapeaux(SK, m=M_REF):
    """+1 si SK > 1+3σ, -1 si SK < 1-3σ, 0 sinon (σ = 2/sqrt(M))."""
    lo, hi = seuils(m)
    return np.where(SK > hi, 1, np.where(SK < lo, -1, 0))


def sk_glissant(p, m, pas):
    """SK d'une série de puissances sur fenêtre glissante (sommes cumulées)."""
    c1 = np.concatenate([[0.0], np.cumsum(p)])
    c2 = np.concatenate([[0.0], np.cumsum(p ** 2)])
    deb = np.arange(0, len(p) - m + 1, pas)
    return deb + m / 2, sk(c1[deb + m] - c1[deb], c2[deb + m] - c2[deb], m)


# -------------------------------------------------------------- statistiques ---
def etude_performances(n_real=N_REAL, seed=SEED):
    """Cumule, sur n_real réalisations communes aux 4 scénarios, le nombre de cellules
    signalées et le nombre total par (scénario, état de vérité) ; renvoie aussi, pour
    chaque fenêtre, la fraction de canaux signalés (dans la bande / hors bande)."""
    cpt = {nom: {e: [0, 0] for e in (0, 1, 2, 3)} for nom in NOMS}
    n_fen = N_BLOCKS // M_REF
    par_fen = {nom: {"bande": np.zeros(n_fen), "hors": np.zeros(n_fen)} for nom in NOMS}
    for r in range(n_real):
        c = composantes(np.random.default_rng(seed + 100 + r))
        for nom in NOMS:
            SK = sk_fenetres(puissances(signal_total(c, nom)), M_REF)[2]
            flag = drapeaux(SK) != 0
            e, _ = etats(nom, c["f"])
            par_fen[nom]["bande"] += flag[:, BANDE].mean(axis=1) / n_real
            par_fen[nom]["hors"] += flag[:, HORS].mean(axis=1) / n_real
            for k in (0, 1, 2, 3):
                msk = e == k
                cpt[nom][k][0] += (flag & msk).sum()
                cpt[nom][k][1] += msk.sum()
    return cpt, par_fen


def balayage_niveau_ofdm(seed=SEED):
    """Taux de détection du sursaut (fenêtres avec sursaut) en fonction du niveau OFDM,
    dans la bande OFDM et hors bande. Même réalisation, seule l'amplitude de r change."""
    det = {k: np.zeros((N_REAL_BALAYAGE, len(NIVEAUX_OFDM))) for k in ("bande", "hors")}
    for r in range(N_REAL_BALAYAGE):
        c = composantes(np.random.default_rng(seed + 900 + r))
        _, act = etats("melange", c["f"])
        for i, lv in enumerate(NIVEAUX_OFDM):
            g = 10 ** ((lv - OFDM_DB) / 20)               # l'OFDM est normalisée à OFDM_DB
            SK = sk_fenetres(puissances(signal_total(c, "melange", g)), M_REF)[2]
            fl = drapeaux(SK) != 0
            det["bande"][r, i] = fl[np.ix_(act, BANDE)].mean()
            det["hors"][r, i] = fl[np.ix_(act, HORS)].mean()
    return {k: (v.mean(axis=0), v.std(axis=0)) for k, v in det.items()}


# ------------------------------------------------------------------ graphiques ---
def fig_quatre_scenarios(P, SKw, f, chemin):
    n_fen = N_BLOCKS // M_REF
    t_ms = np.arange(N_BLOCKS) * DT * 1e3
    deb, fin = periode_sursaut()
    tdeb, tfin = deb * DT * 1e3, fin * DT * 1e3
    lo, hi = seuils(M_REF)
    t0 = np.arange(n_fen) * T_FEN_MS
    cols = plt.cm.viridis(np.linspace(0, 0.85, len(W_SHOW)))
    en_sursaut = (np.arange(N_BLOCKS) >= deb) & (np.arange(N_BLOCKS) < fin)
    fig = plt.figure(figsize=(20, 21))
    gs = GridSpec(5, 4, figure=fig, hspace=0.5, wspace=0.27)
    for j, nom in enumerate(NOMS):
        has_b, has_o = nom in A_SURSAUT_ON, nom in OFDM_ON
        # (1) spectrogramme : moyenne par paquets de 32 spectres, en dB / bruit
        ax = fig.add_subplot(gs[0, j])
        pq = 32
        Pm = P[nom][:N_BLOCKS // pq * pq].reshape(-1, pq, NFFT).mean(axis=1)
        im = ax.imshow(10 * np.log10(Pm.T / SIGMA2), origin="lower", aspect="auto", cmap="viridis",
                       extent=[0, N_BLOCKS * DT * 1e3, F_MHZ[0], F_MHZ[-1]], vmin=-1, vmax=10)
        ax.set_title(TITRES[nom], fontsize=12)
        ax.set_xlabel("temps (ms)")
        ax.set_ylabel("fréquence relative (MHz)")
        if has_b:
            for t in (tdeb, tfin):
                ax.axvline(t, color="w", ls="--", lw=1.3)
            ax.text((tdeb + tfin) / 2, F_MHZ[-1] - 0.8, "sursaut", color="w", ha="center", va="top",
                    fontsize=9, fontweight="bold")
        if has_o:
            for fb in (-BANDE_MHZ, BANDE_MHZ):
                ax.axhline(fb, color="w", ls=":", lw=1.3)
            ax.text(N_BLOCKS * DT * 1e3 * 0.99, BANDE_MHZ + 0.4, "bande OFDM (±9 MHz)", color="w",
                    ha="right", fontsize=8)
        if j == 3:
            fig.colorbar(im, ax=ax, pad=0.02).set_label("puissance (dB / bruit)")
        # (2) spectre de puissance moyenne
        ax = fig.add_subplot(gs[1, j])
        ax.plot(F_MHZ, 10 * np.log10(P[nom].mean(axis=0) / SIGMA2), color=COUL[nom], lw=1.8,
                label="moyenne sur toute l'observation")
        ax.plot(F_MHZ, 10 * np.log10(P[nom][en_sursaut].mean(axis=0) / SIGMA2), color=COUL[nom], lw=1.2,
                ls="--", label="moyenne pendant la période du sursaut")
        ax.axhline(0, color="k", ls=":", lw=0.8)
        if has_o:
            for fb in (-BANDE_MHZ, BANDE_MHZ):
                ax.axvline(fb, color="k", ls=":", lw=0.8)
        ax.set_ylim(-1, 11)
        ax.set_xlabel("fréquence relative (MHz)")
        ax.set_ylabel("puissance moyenne (dB / bruit)")
        ax.legend(fontsize=7, loc="upper right")
        # (3) SK en fonction de la fréquence, plusieurs fenêtres
        ax = fig.add_subplot(gs[2, j])
        ax.axhspan(lo, hi, color="k", alpha=0.09)
        for w, c in zip(W_SHOW, cols):
            act_w = f[w * M_REF:(w + 1) * M_REF].max() > BURST_ACTIVE_FRAC
            ax.plot(F_MHZ, SKw[nom][w], color=c, lw=1.8 if (act_w and has_b) else 0.9,
                    label=f"fenêtre {w} : {t0[w]:.1f}–{t0[w] + T_FEN_MS:.1f} ms" + (" (sursaut)" if act_w else ""))
        ax.axhline(1, color="k", ls=":", lw=0.8)
        if has_o:
            for fb in (-BANDE_MHZ, BANDE_MHZ):
                ax.axvline(fb, color="k", ls=":", lw=0.8)
        ax.set_ylim(0.3, 2.8)
        ax.set_xlabel("fréquence relative (MHz)")
        ax.set_ylabel(f"SK (M = {M_REF})")
        if j == 0:
            ax.legend(fontsize=6.5, loc="upper left")
        ax.set_title("bande grise = 1 ± 3σ ; épais = fenêtre avec sursaut", fontsize=8)
        # (4) puissance dans un canal de la bande OFDM
        ax = fig.add_subplot(gs[3, j])
        ax.plot(t_ms[64:-64:8], 10 * np.log10(lissage(P[nom][:, CAN_BANDE], 64)[64:-64:8] / SIGMA2),
                color=COUL[nom], lw=1.2)
        if has_b:
            ax.axvspan(tdeb, tfin, color="orange", alpha=0.18, label="période du sursaut")
        for k in range(1, n_fen):
            ax.axvline(k * T_FEN_MS, color="gray", lw=0.4, alpha=0.5)
        ax.set_ylim(-2, 12)
        ax.set_xlabel("temps (ms)")
        ax.set_ylabel("puissance (dB / bruit)")
        ax.set_title(f"canal {F_MHZ[CAN_BANDE]:+.2f} MHz (moyenne glissante 64 spectres)", fontsize=9)
        if has_b:
            ax.legend(fontsize=7, loc="upper right")
        # (5) SK de ce canal au cours du temps
        ax = fig.add_subplot(gs[4, j])
        tc, skg = sk_glissant(P[nom][:, CAN_BANDE], M_REF, 64)
        ax.axhspan(lo, hi, color="k", alpha=0.09)
        ax.plot(tc * DT * 1e3, skg, color=COUL[nom], lw=1.2, label="fenêtre glissante (pas 64 spectres)")
        tw = (np.arange(n_fen) + 0.5) * T_FEN_MS
        ax.plot(tw, SKw[nom][:, CAN_BANDE], "o", color="k", ms=4, label="fenêtres disjointes")
        if has_b:
            ax.axvspan(tdeb, tfin, color="orange", alpha=0.18)
        ax.axhline(1, color="k", ls=":", lw=0.8)
        ax.set_ylim(0.5, 2.6)
        ax.set_xlabel("temps (ms)")
        ax.set_ylabel(f"SK du canal (M = {M_REF})")
        ax.set_title(f"SK du canal {F_MHZ[CAN_BANDE]:+.2f} MHz", fontsize=9)
        ax.legend(fontsize=7, loc="upper right")
    fig.suptitle(f"Clear Sky - quatre scénarios sur les mêmes réalisations   "
                 f"(FS = {FS / 1e6:.2f} MHz, NFFT = {NFFT}, canal = {FS / NFFT / 1e3:.0f} kHz, "
                 f"1 spectre = {DT * 1e6:.2f} µs, M = {M_REF} = {T_FEN_MS:.2f} ms ; OFDM = {OFDM_DB:.0f} dB, "
                 f"sursaut au pic = {BURST_PEAK_DB:.0f} dB)", fontsize=12, y=0.915)
    fig.savefig(chemin, dpi=90, bbox_inches="tight")
    plt.close(fig)


def fig_comparaison(SKw, f, bal, chemin):
    n_fen = N_BLOCKS // M_REF
    lo, hi = seuils(M_REF)
    tw = (np.arange(n_fen) + 0.5) * T_FEN_MS
    deb, fin = periode_sursaut()
    tdeb, tfin = deb * DT * 1e3, fin * DT * 1e3
    mu = SIGMA2 * (1 + (10 ** (BURST_PEAK_DB / 10) - 1) * f)
    R = 10 ** (OFDM_DB / 10) * SIGMA2
    th_mel = sk_theorique(mu, M_REF, extra=R)
    th_sur = sk_theorique(mu, M_REF)
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    for a, (masque, titre) in zip(ax[0], [(BANDE, "(a) canaux DANS la bande OFDM"),
                                          (HORS, "(b) canaux HORS de la bande OFDM")]):
        nc = masque.sum()
        s = 2 / np.sqrt(M_REF) / np.sqrt(nc)
        a.axhspan(1 - 3 * s, 1 + 3 * s, color="k", alpha=0.1, label="1 ± 3σ de la moyenne des canaux")
        a.axvspan(tdeb, tfin, color="orange", alpha=0.18, label="période du sursaut")
        for nom in NOMS:
            a.plot(tw, SKw[nom][:, masque].mean(axis=1), "o-", color=COUL[nom], ms=4, lw=1.1,
                   label=TITRES[nom][3:])
        a.set_xlabel("temps (ms)")
        a.set_ylabel(f"SK moyenne sur {nc} canaux (M = {M_REF})")
        a.set_title(titre, fontsize=11)
        a.set_ylim(0.9, 1.7)
        a.legend(fontsize=7, loc="upper right")
    # (c) additivité de la SK ?
    a = ax[1, 0]
    ws = [w for w in range(n_fen) if f[w * M_REF:(w + 1) * M_REF].max() > BURST_ACTIVE_FRAC]
    x = np.arange(len(ws))
    sb = np.array([SKw["sursaut"][w, BANDE].mean() for w in ws])
    so = np.array([SKw["ofdm"][w, BANDE].mean() for w in ws])
    sm = np.array([SKw["melange"][w, BANDE].mean() for w in ws])
    larg = 0.16
    a.bar(x - 2 * larg, sb, larg, color="tab:orange", label="sursaut seul")
    a.bar(x - larg, so, larg, color="tab:blue", label="OFDM seule")
    a.bar(x, sb + so - 1, larg, color="lightgray", edgecolor="k", hatch="//",
          label="'addition' naïve  SK_sursaut + SK_OFDM − 1  (FAUX)")
    a.bar(x + larg, sm, larg, color="tab:red", label="mélange : SK recalculée sur le signal total")
    a.bar(x + 2 * larg, th_mel[ws], larg, color="none", edgecolor="tab:red", hatch="..",
          label="formule  2<μ²>/<μ>² − 1  avec μ = puissance totale")
    a.axhline(hi, color="k", ls="--", lw=0.8)
    a.axhline(1, color="k", ls=":", lw=0.8)
    a.set_xticks(x)
    a.set_xticklabels([f"fenêtre {w}\n{w * T_FEN_MS:.1f}–{(w + 1) * T_FEN_MS:.1f} ms" for w in ws])
    a.set_ylabel("SK moyenne dans la bande OFDM")
    a.set_title("(c) la SK ne s'additionne pas (fenêtres avec sursaut, canaux de la bande)", fontsize=11)
    a.set_ylim(0.8, 1.9)
    a.legend(fontsize=7, loc="upper right")
    # (d) masquage du sursaut par l'OFDM
    a = ax[1, 1]
    for k, c, lab in [("hors", "tab:orange", "hors bande OFDM"), ("bande", "tab:red", "dans la bande OFDM")]:
        m_, s_ = bal[k]
        a.errorbar(NIVEAUX_OFDM, m_ * 100, yerr=s_ * 100, fmt="o-", color=c, capsize=3, label=lab)
    a.axvline(OFDM_DB, color="k", ls=":", label=f"niveau utilisé ({OFDM_DB:.0f} dB)")
    a.set_xlabel("puissance OFDM dans un canal occupé (dB / bruit)")
    a.set_ylabel("taux de détection du sursaut (%)")
    a.set_title(f"(d) l'OFDM masque le sursaut dans sa bande ({N_REAL_BALAYAGE} réalisations)", fontsize=11)
    a.set_ylim(-3, 105)
    a.legend(fontsize=8)
    fig.suptitle(f"Comparaison SK : sursaut seul, OFDM seule, mélange (SK recalculée sur le signal total, M = {M_REF})",
                 fontsize=12)
    plt.tight_layout()
    fig.savefig(chemin, dpi=95, bbox_inches="tight")
    plt.close(fig)


def fig_performance(SKw, f, cpt, chemin):
    n_fen = N_BLOCKS // M_REF
    t0 = np.arange(n_fen) * T_FEN_MS
    deb, fin = periode_sursaut()
    fig = plt.figure(figsize=(18, 11))
    gs = GridSpec(2, 4, figure=fig, height_ratios=[1.15, 1], hspace=0.38, wspace=0.28)
    act = f[:n_fen * M_REF].reshape(n_fen, M_REF).max(axis=1) > BURST_ACTIVE_FRAC
    w_act = np.where(act)[0]
    for j, nom in enumerate(NOMS):
        ax = fig.add_subplot(gs[0, j])
        d = drapeaux(SKw[nom])
        ax.imshow(d, origin="lower", aspect="auto", cmap="bwr", vmin=-1.5, vmax=1.5,
                  extent=[F_MHZ[0], F_MHZ[-1], 0, n_fen * T_FEN_MS])
        if nom in OFDM_ON:
            for fb in (-BANDE_MHZ, BANDE_MHZ):
                ax.axvline(fb, color="k", ls=":", lw=1.2)
        if nom in A_SURSAUT_ON:
            ax.axhline(w_act[0] * T_FEN_MS, color="k", ls="--", lw=1.2)
            ax.axhline((w_act[-1] + 1) * T_FEN_MS, color="k", ls="--", lw=1.2)
        ax.set_title(TITRES[nom], fontsize=11)
        ax.set_xlabel("fréquence relative (MHz)")
        ax.set_ylabel("début de la fenêtre (ms)")
        if j == 0:
            ax.text(0.02, 0.02, "rouge : SK > 1+3σ\nbleu : SK < 1−3σ\nblanc : pas d'anomalie\n"
                    "pointillés : bande OFDM\ntirets : fenêtres avec sursaut", transform=ax.transAxes,
                    fontsize=7, va="bottom", bbox=dict(fc="w", alpha=0.85, ec="gray"))
    # taux par état de vérité
    ax = fig.add_subplot(gs[1, :2])
    etats_aff = [0, 1, 2, 3]
    larg = 0.2
    for i, nom in enumerate(NOMS):
        vals = [100 * cpt[nom][k][0] / cpt[nom][k][1] if cpt[nom][k][1] and
                (k == 0 or (k in (1, 3) and nom in A_SURSAUT_ON) or (k in (2, 3) and nom in OFDM_ON))
                else np.nan for k in etats_aff]
        # pour "bruit" et les états absents, on ne trace que la référence (état 0)
        if nom == "bruit":
            vals = [100 * cpt[nom][0][0] / cpt[nom][0][1]] + [np.nan] * 3
            vals = [100 * sum(cpt[nom][k][0] for k in range(4)) / sum(cpt[nom][k][1] for k in range(4))] + [np.nan] * 3
        bars = ax.bar(np.arange(4) + (i - 1.5) * larg, vals, larg, color=COUL[nom], label=TITRES[nom][3:])
        for b, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}", ha="center", fontsize=7)
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels([f"{ETAT_NOM[k]}\n" + ("(fausses alarmes)" if k == 0 else "(détection)") for k in etats_aff])
    ax.set_ylabel("cellules (fenêtre, canal) signalées (%)")
    ax.set_title(f"Taux de signalement selon ce qui est réellement présent ({N_REAL} réalisations, seuils 3σ)",
                 fontsize=10)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8, loc="upper left")
    # origine des anomalies du mélange
    ax = fig.add_subplot(gs[1, 2:])
    fl = np.array([cpt["melange"][k][0] for k in etats_aff], float)
    part = fl / fl.sum() * 100
    coul = ["lightgray", "tab:orange", "tab:blue", "tab:red"]
    bas = 0
    for k in etats_aff:
        ax.barh(0, part[k], left=bas, color=coul[k], edgecolor="k",
                label=f"{ETAT_NOM[k]} : {part[k]:.1f} %")
        bas += part[k]
    sans_rfi = part[0] + part[1]
    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_xlabel("part des cellules signalées dans le scénario 4 (%)")
    ax.set_title(f"D'où viennent les anomalies du mélange ? {sans_rfi:.1f} % surviennent là où AUCUNE RFI n'est présente",
                 fontsize=10)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2)
    fig.suptitle("Anomalies SK (M = 1024) : signalement selon la vérité terrain", fontsize=12)
    fig.savefig(chemin, dpi=95, bbox_inches="tight")
    plt.close(fig)
    return part


# ------------------------------------------------------------------------ main ---
if __name__ == "__main__":
    dossier = "/home/claude/separation-algorithms-/clear_sky/"
    n_fen = N_BLOCKS // M_REF
    lo, hi = seuils(M_REF)
    c0 = composantes(np.random.default_rng(SEED))
    f = c0["f"]
    P = {nom: puissances(signal_total(c0, nom)) for nom in NOMS}
    SKw = {nom: sk_fenetres(P[nom], M_REF)[2] for nom in NOMS}
    deb, fin = periode_sursaut()
    print(f"Seuils M={M_REF} : [{lo:.3f} ; {hi:.3f}] ; sursaut de {deb * DT * 1e3:.1f} à {fin * DT * 1e3:.1f} ms ;"
          f" bande OFDM = {BANDE.sum()} canaux, hors bande = {HORS.sum()} canaux")

    # --- vérification : puissance du mélange = somme des puissances (composantes indépendantes) ---
    pm = {nom: P[nom].mean() for nom in NOMS}
    print("Puissance moyenne (u.a.) : bruit %.3f ; sursaut %.3f ; OFDM %.3f ; mélange %.3f ; "
          "bruit + excès sursaut + excès OFDM = %.3f" % (pm["bruit"], pm["sursaut"], pm["ofdm"], pm["melange"],
                                                       pm["bruit"] + (pm["sursaut"] - pm["bruit"]) + (pm["ofdm"] - pm["bruit"])))

    # --- table par fenêtre : vérité terrain et canaux signalés ---
    print("\nFenêtre | temps (ms)  | sursaut | OFDM | signalés dans la bande / hors bande (sur 37 / 23 canaux)")
    print("        |             |         |      | " + " | ".join(f"{TITRES[n][3:20]:>20s}" for n in NOMS))
    _, act = etats("melange", f)
    for w in range(n_fen):
        cell = []
        for nom in NOMS:
            d = drapeaux(SKw[nom][w]) != 0
            cell.append(f"{d[BANDE].sum():2d} / {d[HORS].sum():2d}".rjust(20))
        print(f"  {w:2d}    | {w * T_FEN_MS:4.1f}–{(w + 1) * T_FEN_MS:4.1f} |   {'oui' if act[w] else 'non'}   | oui  | "
              + " | ".join(cell))

    # --- additivité de la SK : mesure vs addition naïve vs formule ---
    mu = SIGMA2 * (1 + c0["a"] * f)
    R = 10 ** (OFDM_DB / 10) * SIGMA2
    th = sk_theorique(mu, M_REF, extra=R)
    print("\nFenêtres avec sursaut, SK moyenne dans la bande OFDM :")
    print("  fen | sursaut | OFDM  | naïf S+O-1 | mélange mesuré | formule | sursaut seul hors bande | mélange hors bande")
    for w in np.where(act)[0]:
        sb, so, sm = (SKw[k][w, BANDE].mean() for k in ("sursaut", "ofdm", "melange"))
        print(f"  {w:3d} | {sb:7.3f} | {so:5.3f} | {sb + so - 1:10.3f} | {sm:14.3f} | {th[w]:7.3f} | "
              f"{SKw['sursaut'][w, HORS].mean():23.3f} | {SKw['melange'][w, HORS].mean():.3f}")

    # --- statistiques sur N_REAL réalisations ---
    cpt, par_fen = etude_performances()
    print(f"\nTaux de signalement (%) sur {N_REAL} réalisations, seuils 3σ (cellules : fenêtre x canal)")
    print(f"{'scénario':28s}" + "".join(f"{ETAT_NOM[k]:>22s}" for k in range(4)))
    for nom in NOMS:
        ligne = f"{TITRES[nom][3:]:28s}"
        for k in range(4):
            n_ok, n_tot = cpt[nom][k]
            ligne += (f"{100 * n_ok / n_tot:15.2f} (N={n_tot:5d})" if n_tot and
                      ((k == 0) or (k in (1, 3) and nom in A_SURSAUT_ON) or (k in (2, 3) and nom in OFDM_ON))
                      else " " * 22)
        print(ligne)
    tot_n = sum(cpt["bruit"][k][1] for k in range(4))
    tot_f = sum(cpt["bruit"][k][0] for k in range(4))
    print(f"Référence : fausses alarmes du bruit seul (toutes cellules) = {100 * tot_f / tot_n:.2f} %  "
          f"(approximation gaussienne 3σ : 0,27 %)")

    print(f"\nFraction de canaux signalés (%) par fenêtre avec sursaut, moyenne sur {N_REAL} réalisations")
    print("  fen | sursaut: bande / hors | mélange: bande / hors | OFDM seule: bande")
    for w in np.where(act)[0]:
        print(f"  {w:3d} | {par_fen['sursaut']['bande'][w] * 100:11.1f} / {par_fen['sursaut']['hors'][w] * 100:4.1f}"
              f"   | {par_fen['melange']['bande'][w] * 100:11.1f} / {par_fen['melange']['hors'][w] * 100:4.1f}"
              f"   | {par_fen['ofdm']['bande'][w] * 100:8.1f}")

    bal = balayage_niveau_ofdm()
    print("\nDétection du sursaut (%) selon le niveau OFDM : niveau | dans la bande | hors bande")
    for i, lv in enumerate(NIVEAUX_OFDM):
        print(f"   {lv:4d} dB | {bal['bande'][0][i] * 100:6.1f} ± {bal['bande'][1][i] * 100:4.1f} | "
              f"{bal['hors'][0][i] * 100:6.1f} ± {bal['hors'][1][i] * 100:4.1f}")

    fig_quatre_scenarios(P, SKw, f, dossier + "sk_melange_quatre_scenarios.png")
    fig_comparaison(SKw, f, bal, dossier + "sk_melange_comparaison.png")
    part = fig_performance(SKw, f, cpt, dossier + "sk_melange_performance.png")
    print("\nOrigine des anomalies du mélange (%%) : %s" % ", ".join(
        f"{ETAT_NOM[k]} {part[k]:.1f}" for k in range(4)))
    print("Figures : sk_melange_quatre_scenarios.png, sk_melange_comparaison.png, sk_melange_performance.png")
