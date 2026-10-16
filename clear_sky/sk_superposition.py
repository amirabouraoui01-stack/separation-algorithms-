"""
Clear Sky - Superposition sursaut + RFI OFDM : quand la SK seule ne suffit pas.

Réutilise sk_scenarios.py (mêmes modèles, mêmes unités : bruit = 0 dB, 1 spectre = DT).
Ce script ne sert qu'APRÈS l'analyse séparée des scénarios : on y étudie
  (a) les spectres de SK de quatre cas, dont sursaut + OFDM superposés ;
  (b) la "dilution" de la signature du sursaut par une RFI OFDM gaussienne continue :
      simulation vs formule  SK ~ 2<mu^2>/<mu>^2 - 1  avec mu = puissance totale ;
  (c) l'ambiguïté : sursaut et OFDM intermittent donnent des SK comparables ;
  (d) ce qui les distingue malgré tout : forme temporelle et occupation spectrale.
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sk_scenarios as sc
from sk_scenarios import (NFFT, N_BLOCKS, SIGMA2, M_REF, DT, FS, SUBFRAME, F_MHZ, T_FEN_MS,
                          CAN_CENTRE, bruit_complexe, ofdm_lte, scenario_sursaut, puissances,
                          sk_fenetres, seuils, sk_theorique, lissage)

N_REAL = 4   # réalisations pour la courbe de dilution


def scenario_superpose(rng, intermittent=False, level_db=None):
    """Bruit + émission solaire gaussienne (profil sursaut) + RFI OFDM, tous additifs.
    Retourne le signal, le profil du sursaut f, et le coefficient a du sursaut."""
    x, f, a = scenario_sursaut(rng)            # bruit récepteur + sursaut
    n = len(x)
    active = None
    if intermittent:
        active = rng.random(int(np.ceil(n / SUBFRAME))) < sc.OFDM_ON_PROB
    return x + ofdm_lte(n, rng, active, level_db=level_db), f, a


def fenetre_sursaut(P_sursaut):
    """Fenêtre de M_REF spectres où le sursaut modifie le plus la SK."""
    _, _, SK = sk_fenetres(P_sursaut, M_REF)
    return int(np.argmax(SK.mean(axis=1)))


if __name__ == "__main__":
    dossier = "/home/claude/separation-algorithms-/clear_sky/"
    rng = np.random.default_rng(sc.SEED)
    n = N_BLOCKS * NFFT
    d = np.abs(np.arange(NFFT) - NFFT // 2)
    bande = d <= 18                           # canaux de la bande OFDM occupée
    lo, hi = seuils(M_REF)
    n_fen = N_BLOCKS // M_REF
    R = 10 ** (sc.OFDM_DB / 10) * SIGMA2      # puissance OFDM par canal occupé (u.a.)

    # --- quatre cas, mêmes tirages de bruit pour la comparaison ---
    seed = 123
    x_s, f, a = scenario_sursaut(np.random.default_rng(seed))
    cas = {"sursaut seul": puissances(x_s)}
    cas["OFDM charge pleine seul"] = puissances(
        bruit_complexe(n, SIGMA2, np.random.default_rng(seed + 1)) + ofdm_lte(
            n, np.random.default_rng(seed + 2)))
    r = np.random.default_rng(seed + 3)
    act = r.random(int(np.ceil(n / SUBFRAME))) < sc.OFDM_ON_PROB
    cas["OFDM intermittent seul"] = puissances(bruit_complexe(n, SIGMA2, r) + ofdm_lte(n, r, act))
    x_sup, _, _ = scenario_superpose(np.random.default_rng(seed + 4))
    cas["sursaut + OFDM charge pleine"] = puissances(x_sup)

    w = fenetre_sursaut(cas["sursaut seul"])
    sks = {k: sk_fenetres(P, M_REF)[2] for k, P in cas.items()}
    mu = SIGMA2 * (1 + a * f)

    print(f"Fenêtre étudiée : {w} ({w * T_FEN_MS:.1f}–{(w + 1) * T_FEN_MS:.1f} ms), M = {M_REF}, "
          f"seuils [{lo:.3f} ; {hi:.3f}], OFDM = {sc.OFDM_DB:.0f} dB / bruit")
    print(f"{'cas':32s} {'SK dans la bande':>17s} {'SK hors bande':>14s}")
    for k, SK in sks.items():
        print(f"{k:32s} {SK[w, bande].mean():17.3f} {SK[w, d >= 21].mean():14.3f}")
    th_sup = sk_theorique(mu, M_REF, extra=R)[w]
    th_seul = sk_theorique(mu, M_REF)[w]
    print(f"Théorie fenêtre {w} : sursaut seul {th_seul:.3f} ; sursaut + OFDM (dans la bande) {th_sup:.3f}")

    # --- (b) dilution : SK dans la bande vs niveau OFDM ---
    niveaux = np.arange(-10, 21, 5)
    sim, th = [], []
    for lv in niveaux:
        vals = []
        for k in range(N_REAL):
            xs, ff, aa = scenario_superpose(np.random.default_rng(500 + k), level_db=lv)
            P = puissances(xs)
            SKk = sk_fenetres(P, M_REF)[2]
            vals.append(SKk[w, bande].mean())
        sim.append((np.mean(vals), np.std(vals)))
        th.append(sk_theorique(mu, M_REF, extra=10 ** (lv / 10) * SIGMA2)[w])
    print("Dilution (SK dans la bande, fenêtre %d) : niveau OFDM (dB) / simulation / théorie" % w)
    for lv, (m_, s_), t_ in zip(niveaux, sim, th):
        print(f"   {lv:4d} dB : {m_:.3f} ± {s_:.3f}   théorie {t_:.3f}")

    # --- figure ---
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    coul = {"sursaut seul": "tab:orange", "OFDM charge pleine seul": "tab:blue",
            "OFDM intermittent seul": "tab:green", "sursaut + OFDM charge pleine": "tab:red"}
    a0 = ax[0, 0]
    a0.axhspan(lo, hi, color="k", alpha=0.08, label=f"1 ± 3σ (M={M_REF})")
    a0.axvspan(F_MHZ[CAN_CENTRE - 18], F_MHZ[CAN_CENTRE + 18], color="tab:blue", alpha=0.07,
               label="bande OFDM occupée")
    for k, SK in sks.items():
        a0.plot(F_MHZ, SK[w], color=coul[k], lw=1.6, label=k)
    a0.set_ylim(0.5, 2.6)
    a0.set_xlabel("fréquence relative (MHz)")
    a0.set_ylabel(f"SK (M = {M_REF})")
    a0.set_title(f"(a) SK par canal, fenêtre {w} ({w * T_FEN_MS:.1f}–{(w + 1) * T_FEN_MS:.1f} ms)", fontsize=10)
    a0.legend(fontsize=7, loc="upper right")

    a1 = ax[0, 1]
    a1.axhline(1, color="k", ls=":", lw=0.8)
    a1.axhline(hi, color="k", ls="--", lw=0.8, label=f"seuil 1 + 3σ (M={M_REF})")
    a1.errorbar(niveaux, [m_ for m_, _ in sim], yerr=[s_ for _, s_ in sim], fmt="o", color="tab:red",
                capsize=3, label=f"simulation ({N_REAL} réalisations, bande OFDM)")
    a1.plot(niveaux, th, "-", color="tab:red", alpha=0.6, label=r"théorie $2\langle\mu^2\rangle/\langle\mu\rangle^2-1$")
    a1.axhline(th_seul, color="tab:orange", ls="--", label="sursaut seul (hors bande)")
    a1.set_xlabel("puissance OFDM dans un canal (dB / bruit)")
    a1.set_ylabel(f"SK dans la bande, fenêtre {w}")
    a1.set_title("(b) une RFI gaussienne stationnaire dilue la signature du sursaut", fontsize=10)
    a1.legend(fontsize=8, loc="center right")

    a2 = ax[1, 0]
    t_fen = (np.arange(n_fen) + 0.5) * T_FEN_MS
    sk_sur = sks["sursaut seul"].mean(axis=1)
    sk_int = sks["OFDM intermittent seul"][:, bande].mean(axis=1)
    a2.axhspan(lo, hi, color="k", alpha=0.08, label=f"1 ± 3σ (M={M_REF})")
    a2.plot(t_fen, sk_sur, "o-", color="tab:orange", label="sursaut (moyenne des canaux)")
    a2.plot(t_fen, sk_int, "s-", color="tab:green", label="OFDM intermittent (canaux de la bande)")
    a2.set_xlabel("temps (ms)")
    a2.set_ylabel(f"SK (M = {M_REF})")
    a2.set_title("(c) des SK comparables : la SK seule ne dit pas l'origine", fontsize=10)
    a2.legend(fontsize=8)

    a3 = ax[1, 1]
    t_ms = np.arange(N_BLOCKS) * DT * 1e3
    for k, c in [("sursaut seul", "tab:orange"), ("OFDM intermittent seul", "tab:green")]:
        a3.plot(t_ms[64:-64:8], 10 * np.log10(lissage(cas[k][:, CAN_CENTRE], 64)[64:-64:8] / SIGMA2),
                color=c, lw=1.2, label=f"{k}, canal {F_MHZ[CAN_CENTRE]:+.1f} MHz")
    a3.plot(t_ms[64:-64:8], 10 * np.log10(lissage(cas["OFDM intermittent seul"][:, CAN_CENTRE + 24], 64)
                                          [64:-64:8] / SIGMA2), color="tab:green", ls="--", lw=1,
            label=f"OFDM intermittent, canal {F_MHZ[CAN_CENTRE + 24]:+.1f} MHz (hors bande)")
    for k in range(1, n_fen):
        a3.axvline(k * T_FEN_MS, color="gray", lw=0.4, alpha=0.5)
    a3.set_ylim(-2, 11)
    a3.set_xlabel("temps (ms)")
    a3.set_ylabel("puissance (dB / bruit)")
    a3.set_title("(d) ce qui les distingue : forme temporelle et occupation spectrale", fontsize=10)
    a3.legend(fontsize=7, loc="upper right")
    fig.suptitle(f"Superposition et limites de la SK seule  (NFFT = {NFFT}, canal = {FS / NFFT / 1e3:.0f} kHz, "
                 f"1 spectre = {DT * 1e6:.2f} µs, M = {M_REF} = {T_FEN_MS:.2f} ms)", fontsize=11)
    plt.tight_layout()
    fig.savefig(dossier + "sk_superposition.png", dpi=100, bbox_inches="tight")
    print("Figure : sk_superposition.png")
