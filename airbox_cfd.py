#!/usr/bin/env python3
"""
airbox_cfd.py  -  Stroemungsanalyse fuer Luftfilterkaesten (Rohluftseite)
==========================================================================

Was das Tool macht
------------------
1. Liest ein STL des Kasten-Unterteils (ein geschlossener Koerper, Einheit mm).
2. Zerlegt es in Wuerfelzellen (Voxel) und findet den Innenraum ueber den Einlass.
3. Verlaengert die Einlassoeffnung um ein kurzes Rohr und gibt dort die
   Einstroemgeschwindigkeit vor.
4. Modelliert den Filter als poroese Schicht (Seal-Ebene bis Faltenunterkante)
   mit einstellbarem Druckverlust. Quer zu den Falten ist der Widerstand hoeher.
5. Ueber dem Filter: gleichmaessiger Druckauslass (Reinluftseite wird nicht
   modelliert, damit nur die Rohluftseite bewertet wird).
6. Loest die Stroemung mit der Lattice-Boltzmann-Methode (D3Q19, BGK,
   Smagorinsky-Turbulenzmodell) und mittelt die Ergebnisse zeitlich.
7. Gibt aus:
   - Gleichfoermigkeitsindex gamma der Filteranstroemung (1 = perfekt gleichmaessig)
   - Flaechenanteile unterversorgt (< 50 % des Mittels) und Strahlzonen (> 150 %)
   - Karte der Filteranstroemung und Geschwindigkeitsschnitte als PNG
   - Kennwerte als JSON

Voraussetzungen an das Modell
-----------------------------
- Ein geschlossener Koerper (wasserdichtes STL), Einheit mm.
- Filterebene senkrecht zu einer Koordinatenachse (--depth-axis), Lage per --seal.
- Ein Einlass auf einer der aeusseren Seiten (--inlet-face), als Loch in einer
  flachen Wand oder als vorstehender Stutzen. Mehrere Oeffnungen auf dieser
  Seite werden alle als Einlass mit gleicher Geschwindigkeit behandelt.
- Unten offene Taschen (Schalenboden) sind erlaubt, sie werden geschlossen.

Was das Tool NICHT kann
-----------------------
- Die echte Reynoldszahl (~1e5) wird nicht erreicht. Die Stroemung wird bei
  deutlich kleinerer Reynoldszahl mit Turbulenzmodell gerechnet. Ergebnisse
  sind qualitativ: gut fuer den VERGLEICH von Varianten, nicht fuer exakte
  Druckverluste in mbar.
- Bauteile duenner als etwa eine Zellgroesse werden beim Voxelisieren
  aufgedickt (Rippen 2,5 mm werden bei dx = 3 mm zu 3-6 mm).

Benutzung
---------
    pip install numpy matplotlib
    python airbox_cfd.py Koerper1.stl                 # Standard fuer Alexanders Modell
    python airbox_cfd.py variante.stl --tag v2        # Variante, eigener Dateiname
    python airbox_cfd.py k.stl --dx 2.5 --steps 8000  # feiner, dauert laenger

Achsen (Standard passend zu Koerper1.stl):
    --depth-axis y --depth-sign +   Tiefe laeuft entlang +Y (Flansch oben bei Y=0, Boden bei Y=150)
    --inlet-face zmin               Einlass liegt auf der Seite mit kleinstem Z, Luft stroemt in +Z
    --seal 0 --pleat 40             Dichtebene bei Y=0, Falten 40 mm tief

Laufzeit: dx = 3 mm, 6000 Schritte etwa 10-20 Minuten (je nach Rechner).
"""
import argparse, json, math, struct, sys, time
import numpy as np

# ---------------------------------------------------------------- STL
def load_stl(fn):
    data = open(fn, "rb").read()
    if data[:5].lower() == b"solid" and b"facet" in data[:400]:
        pts = [list(map(float, l.split()[1:4])) for l in data.decode(errors="ignore").splitlines()
               if l.strip().startswith("vertex")]
        return np.array(pts, float).reshape(-1, 3, 3)
    n = struct.unpack("<I", data[80:84])[0]
    arr = np.frombuffer(data[84:84 + 50 * n], dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]))
    return arr["v"].reshape(-1, 3, 3).astype(float)

def to_local(T, depth_axis, depth_sign, inlet_face, seal):
    """Koordinaten umsortieren: (X, D, Z) mit D = Tiefe ab Dichtebene (Boden positiv), Einlass bei Z-min."""
    ax = {"x": 0, "y": 1, "z": 2}
    d = ax[depth_axis]; f_ax = ax[inlet_face[0]]
    if f_ax == d: sys.exit("Einlassseite darf nicht auf der Tiefenachse liegen.")
    o = 3 - d - f_ax
    P = T.reshape(-1, 3)
    D = (P[:, d] - seal) * (1 if depth_sign == "+" else -1)
    Z = P[:, f_ax] * (1 if inlet_face.endswith("min") else -1)
    X = P[:, o]
    return np.stack([X, D, Z], 1).reshape(-1, 3, 3)

# ---------------------------------------------------------------- Voxel
def voxelize(T, origin, shape, h):
    """Parity-Voxelisierung entlang Z. Liefert bool[nx,ny,nz] (True = Material)."""
    nx, ny, nz = shape
    hits = [[[] for _ in range(ny)] for _ in range(nx)]
    jit = np.array([1.37e-4, 2.71e-4])
    for tri in T:
        p = tri[:, :2] - origin[:2] - jit
        mn = np.floor(p.min(0) / h - 0.5).astype(int); mx = np.ceil(p.max(0) / h - 0.5).astype(int)
        i0, j0 = max(mn[0], 0), max(mn[1], 0); i1, j1 = min(mx[0], nx - 1), min(mx[1], ny - 1)
        if i1 < i0 or j1 < j0: continue
        I, J = np.meshgrid(np.arange(i0, i1 + 1), np.arange(j0, j1 + 1), indexing="ij")
        cx, cy = (I + 0.5) * h, (J + 0.5) * h
        a, b, c = p
        den = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(den) < 1e-12: continue
        l1 = ((b[1] - c[1]) * (cx - c[0]) + (c[0] - b[0]) * (cy - c[1])) / den
        l2 = ((c[1] - a[1]) * (cx - c[0]) + (a[0] - c[0]) * (cy - c[1])) / den
        l3 = 1 - l1 - l2
        m = (l1 >= 0) & (l2 >= 0) & (l3 >= 0)
        if not m.any(): continue
        z = l1 * tri[0, 2] + l2 * tri[1, 2] + l3 * tri[2, 2] - origin[2]
        for ii, jj, zz in zip(I[m], J[m], z[m]):
            hits[ii][jj].append(zz)
    vox = np.zeros(shape, bool)
    zc = (np.arange(nz) + 0.5) * h
    for i in range(nx):
        for j in range(ny):
            zs = sorted(hits[i][j])
            if len(zs) % 2: zs = zs[:-1]
            for k in range(0, len(zs), 2):
                vox[i, j, (zc >= zs[k]) & (zc < zs[k + 1])] = True
    return vox

def flood(seed, allowed):
    reg = seed & allowed
    while True:
        g = reg.copy()
        g[1:] |= reg[:-1]; g[:-1] |= reg[1:]
        g[:, 1:] |= reg[:, :-1]; g[:, :-1] |= reg[:, 1:]
        g[:, :, 1:] |= reg[:, :, :-1]; g[:, :, :-1] |= reg[:, :, 1:]
        g &= allowed
        if g.sum() == reg.sum(): return reg
        reg = g

# ---------------------------------------------------------------- LBM D3Q19
C = np.array([[0,0,0],[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1],
              [1,1,0],[-1,-1,0],[1,-1,0],[-1,1,0],[1,0,1],[-1,0,-1],[1,0,-1],[-1,0,1],
              [0,1,1],[0,-1,-1],[0,1,-1],[0,-1,1]])
W = np.array([1/3] + [1/18]*6 + [1/36]*12, np.float32)
OPP = np.array([0,2,1,4,3,6,5,8,7,10,9,12,11,14,13,16,15,18,17])
Cf = C.astype(np.float32)

def feq(rho, ux, uy, uz):
    usq = 1.5 * (ux*ux + uy*uy + uz*uz)
    out = np.empty((19,) + rho.shape, np.float32)
    for i in range(19):
        cu = 3.0 * (Cf[i,0]*ux + Cf[i,1]*uy + Cf[i,2]*uz)
        out[i] = W[i] * rho * (1.0 + cu + 0.5*cu*cu - usq)
    return out

def main():
    ap = argparse.ArgumentParser(description="Stroemungsanalyse Luftfilterkasten (Rohluftseite)")
    ap.add_argument("stl")
    ap.add_argument("--dx", type=float, default=3.0, help="Zellgroesse in mm (3 = Standard)")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--avg", type=int, default=2000, help="Schritte am Ende fuer die Zeitmittelung")
    ap.add_argument("--depth-axis", default="y"); ap.add_argument("--depth-sign", default="+")
    ap.add_argument("--inlet-face", default="zmin", choices=["xmin","xmax","ymin","ymax","zmin","zmax"])
    ap.add_argument("--seal", type=float, default=0.0, help="Lage der Dichtebene auf der Tiefenachse (mm)")
    ap.add_argument("--pleat", type=float, default=40.0, help="Faltentiefe unter der Dichtebene (mm)")
    ap.add_argument("--K", type=float, default=3.0,
                    help="Filterwiderstand: Druckverlust Filter / Staudruck im Einlass (Papierfilter neu ~2-4)")
    ap.add_argument("--aniso", type=float, default=10.0, help="Widerstand quer zu den Falten (Faktor)")
    ap.add_argument("--u", type=float, default=0.06, help="Einlassgeschwindigkeit in Gittereinheiten")
    ap.add_argument("--tau", type=float, default=0.51, help="Relaxationszeit (Viskositaet), >0.5")
    ap.add_argument("--tag", default="")
    ap.add_argument("--from-npz", default="", help="Nur Auswertung aus einer gespeicherten Rechnung erzeugen")
    a = ap.parse_args()
    tag = ("_" + a.tag) if a.tag else ""
    if a.from_npz:
        d = np.load(a.from_npz)
        acc = {k: d[k] for k in ("ux", "uy", "uz", "rho")}
        hist = d["hist"].tolist() if "hist" in d else []
        report(acc, d["fluid"], d["solid"], d["xc"], d["dc"], d["zc"], a, tag, hist, d["fluid"][:, :, 0], 0)
        return
    t0 = time.time()

    # ---------- Geometrie
    T = to_local(load_stl(a.stl), a.depth_axis, a.depth_sign, a.inlet_face, a.seal)
    P = T.reshape(-1, 3); mn, mx = P.min(0), P.max(0)
    h = a.dx; sub = 3; hf = h / sub
    duct = 10                                          # Zellen Einlassrohr
    origin = np.array([mn[0] - h, mn[1] - h, mn[2] - duct * h])
    shape = tuple(int(math.ceil((mx[i] - origin[i] + h) / h)) for i in range(3))
    print(f"Modell {mx-mn} mm, Gitter {shape} = {np.prod(shape)/1e3:.0f}k Zellen bei dx={h} mm")
    fine = voxelize(T, origin, tuple(s * sub for s in shape), hf)
    solid = fine.reshape(shape[0], sub, shape[1], sub, shape[2], sub).mean((1, 3, 5)) >= 0.25
    nx, ny, nz = shape
    xc = origin[0] + (np.arange(nx) + 0.5) * h
    dc = origin[1] + (np.arange(ny) + 0.5) * h
    zc = origin[2] + (np.arange(nz) + 0.5) * h
    print(f"Voxelisiert in {time.time()-t0:.0f} s")

    # ---------- Einlass: Oeffnung in der Wand bei Z-min
    # Oeffnung = Loch in der vordersten Materialschicht (flache Wand oder Stirnseite eines Stutzens):
    # leere Flaechen, die nicht mit dem Rand des Querschnitts verbunden sind.
    kw0 = int(np.searchsorted(zc, mn[2]))
    kf = int((mn[2] + 1.5 - origin[2]) / hf)
    empty = ~fine[:, :, kf]
    outside = np.zeros_like(empty)
    outside[0, :] = outside[-1, :] = outside[:, 0] = outside[:, -1] = True
    outside &= empty
    while True:
        g = outside.copy()
        g[1:] |= outside[:-1]; g[:-1] |= outside[1:]; g[:, 1:] |= outside[:, :-1]; g[:, :-1] |= outside[:, 1:]
        g &= empty
        if g.sum() == outside.sum(): break
        outside = g
    holes = empty & ~outside
    opening = holes.reshape(nx, sub, ny, sub).mean((1, 3)) >= 0.5
    opening &= (dc[None, :] > 0)                        # nur unterhalb der Dichtebene
    if opening.sum() < 20: sys.exit("Keine Einlassoeffnung gefunden - Achsen/--inlet-face pruefen.")
    solid[:, :, :kw0] = True
    solid[:, :, :kw0][opening] = False                 # Einlassrohr
    solid[:, dc > mx[1] - 0.5 * h, :] = True           # unter dem Boden schliessen (offene Taschen)
    top = int(np.searchsorted(dc, 0.0))                # Auslassebene = erste Zellreihe unter der Dichtebene
    allowed = ~solid; allowed[:, :top, :] = False
    seed = np.zeros(shape, bool); seed[:, :, 0] = opening
    fluid = flood(seed, allowed)
    print(f"Innenraum {fluid.sum()/1e3:.0f}k Zellen ({fluid.sum()*h**3/1e6:.2f} l), Einlass {opening.sum()} Zellen "
          f"({opening.sum()*h*h/100:.0f} cm2)")

    inlet = np.zeros(shape, bool); inlet[:, :, 0] = opening
    outlet = np.zeros(shape, bool); outlet[:, top, :] = fluid[:, top, :]
    porous = fluid & ((dc > dc[top]) & (dc <= a.pleat))[None, :, None]
    A_in = opening.sum(); A_f = fluid[:, int(np.searchsorted(dc, a.pleat / 2)), :].sum()
    u_face = a.u * A_in / A_f
    Lp = max(1, int(((dc > dc[top]) & (dc <= a.pleat)).sum()))
    sig = a.K * 0.5 * a.u**2 / (u_face * Lp)            # Darcy-Koeffizient (Gittereinheiten)
    print(f"Filterflaeche {A_f*h*h/100:.0f} cm2, Flaechenverhaeltnis Filter/Einlass {A_f/A_in:.1f}, sigma={sig:.4f}")

    # ---------- LBM
    rho = np.ones(shape, np.float32); z0 = np.zeros(shape, np.float32)
    f = feq(rho, z0, z0, z0)
    wall = ~fluid
    near = np.zeros(shape, bool)
    for c in C[1:]:
        near |= np.roll(fluid, tuple(-c), (0, 1, 2))
    bb = np.flatnonzero((wall & near).ravel())
    inl = np.flatnonzero(inlet.ravel()); inl_n = np.flatnonzero(np.roll(inlet, 1, 2).ravel())
    out = np.flatnonzero(outlet.ravel()); out_n = np.flatnonzero(np.roll(outlet, 1, 1).ravel())
    pidx = np.flatnonzero(porous.ravel())
    tau0 = a.tau; Cs2 = 0.17**2
    acc = {k: np.zeros(shape, np.float32) for k in ("ux", "uy", "uz", "rho")}
    nacc = 0; hist = []
    tl = time.time()
    for step in range(1, a.steps + 1):
        # Momente
        rho = f.sum(0)
        ux = (f[1] - f[2] + f[7] - f[8] + f[9] - f[10] + f[11] - f[12] + f[13] - f[14]) / rho
        uy = (f[3] - f[4] + f[7] - f[8] - f[9] + f[10] + f[15] - f[16] + f[17] - f[18]) / rho
        uz = (f[5] - f[6] + f[11] - f[12] - f[13] + f[14] + f[15] - f[16] - f[17] + f[18]) / rho
        ux *= ~wall; uy *= ~wall; uz *= ~wall
        fe = feq(rho, ux, uy, uz)
        # Smagorinsky
        neq = f - fe
        Q = np.zeros(shape, np.float32)
        for (p, q) in ((0,0),(1,1),(2,2),(0,1),(0,2),(1,2)):
            Pi = np.tensordot(Cf[:, p] * Cf[:, q], neq, axes=(0, 0))
            Q += (Pi * Pi) * (1.0 if p == q else 2.0)
        tau = 0.5 * (tau0 + np.sqrt(tau0 * tau0 + 18.0 * math.sqrt(2.0) * Cs2 * np.sqrt(Q) / rho))
        # Filter: Geschwindigkeitsverschiebung (Darcy, anisotrop) nur in der poroesen Zone
        tp = tau.ravel()[pidx]
        sx = 1.0 - np.minimum(tp * sig * a.aniso, 0.9); sy = 1.0 - np.minimum(tp * sig, 0.9)
        fe.reshape(19, -1)[:, pidx] = feq(rho.ravel()[pidx], ux.ravel()[pidx] * sx,
                                          uy.ravel()[pidx] * sy, uz.ravel()[pidx] * sx)
        f -= (f - fe) * np.where(wall, np.float32(0), 1.0 / tau).astype(np.float32)
        # Stroemen
        for i in range(1, 19):
            f[i] = np.roll(f[i], tuple(C[i]), (0, 1, 2))
        F = f.reshape(19, -1)
        tmp = F[:, bb]; F[:, bb] = tmp[OPP]               # Wand: Bounce-back
        rn = F[:, inl_n].sum(0)
        F[:, inl] = feq(rn, np.zeros_like(rn), np.zeros_like(rn), np.full_like(rn, a.u))
        G = F[:, out_n]; rg = G.sum(0)
        F[:, out] = feq(np.ones_like(rg),
                        (G[1]-G[2]+G[7]-G[8]+G[9]-G[10]+G[11]-G[12]+G[13]-G[14]) / rg,
                        (G[3]-G[4]+G[7]-G[8]-G[9]+G[10]+G[15]-G[16]+G[17]-G[18]) / rg,
                        (G[5]-G[6]+G[11]-G[12]-G[13]+G[14]+G[15]-G[16]-G[17]+G[18]) / rg)
        if not np.isfinite(F[:, inl[:5]]).all():
            sys.exit(f"Instabil bei Schritt {step} - groessere --tau oder kleinere --u verwenden.")
        if step > a.steps - a.avg:
            acc["ux"] += ux; acc["uy"] += uy; acc["uz"] += uz; acc["rho"] += rho; nacc += 1
        if step % 250 == 0:
            jm = int(np.searchsorted(dc, a.pleat * 0.5))
            w = -uy[:, jm, :][fluid[:, jm, :]]
            g = 1 - np.abs(w - w.mean()).sum() / (2 * w.size * w.mean()) if w.mean() > 0 else float("nan")
            qin = A_in * a.u; qout = -(uy[:, top + 1, :] * fluid[:, top + 1, :]).sum()
            hist.append((step, float(g), float(qout / qin)))
            el = time.time() - tl
            print(f"Schritt {step:6d}  gamma={g:.3f}  Auslass/Einlass={qout/qin:.2f}  "
                  f"{el/step*1000:.0f} ms/Schritt, Rest ~{el/step*(a.steps-step)/60:.1f} min", flush=True)

    for k in acc: acc[k] /= max(nacc, 1)
    np.savez_compressed(f"airbox_result{tag}.npz", fluid=fluid, porous=porous, solid=solid, xc=xc, dc=dc, zc=zc,
                        **acc, u=a.u, pleat=a.pleat, hist=np.array(hist))
    report(acc, fluid, solid, xc, dc, zc, a, tag, hist, opening, top)
    print(f"Fertig in {(time.time()-t0)/60:.1f} min")

# ---------------------------------------------------------------- Auswertung
def report(acc, fluid, solid, xc, dc, zc, a, tag, hist, opening, top):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    u0 = a.u; ux, uy, uz, rho = acc["ux"], acc["uy"], acc["uz"], acc["rho"]
    U = np.sqrt(ux**2 + uy**2 + uz**2) / u0
    ext = [xc[0], xc[-1], zc[0], zc[-1]]

    jm = int(np.searchsorted(dc, a.pleat * 0.5)); m = fluid[:, jm, :]
    w = -uy[:, jm, :]; wm = w[m].mean(); wr = np.where(m, w / wm, np.nan)
    gamma = 1 - np.abs(w[m] - wm).sum() / (2 * m.sum() * wm)
    # Kernbereich ohne die wandnahe Randzelle (dort bremst die Wandhaftung, im echten Filter sitzt der Rahmen)
    core = m.copy()
    core[1:] &= m[:-1]; core[:-1] &= m[1:]; core[:, 1:] &= m[:, :-1]; core[:, :-1] &= m[:, 1:]
    wc = w[core]; gamma_c = 1 - np.abs(wc - wc.mean()).sum() / (2 * wc.size * wc.mean())
    low = float((wc < 0.5 * wc.mean()).mean()); high = float((wc > 1.5 * wc.mean()).mean())
    rev = float((w[m] < 0).mean())
    # Druckverlust Rohluftseite: Einlass bis knapp unter Filter (ohne Filter)
    jb = int(np.searchsorted(dc, a.pleat + 2 * a.dx))
    p_in = rho[:, :, 1][opening].mean() / 3
    p_b = rho[:, jb, :][fluid[:, jb, :]].mean() / 3
    zeta = (p_in - p_b) / (0.5 * u0**2)

    # Einlassmitte (fuer Schnitte)
    ii, jj = np.nonzero(opening); ic, jc = int(ii.mean()), int(jj.mean())
    jlow = min(int(np.searchsorted(dc, a.pleat + 12)), len(dc) - 1)

    fig, ax = plt.subplots(2, 2, figsize=(15, 12.5))
    def plan(axh, data, title, cmap, norm=None, vmax=None, quiv=None):
        im = axh.imshow(data.T, origin="lower", extent=ext, cmap=cmap, norm=norm, vmin=None if norm else 0,
                        vmax=vmax, aspect="equal", interpolation="nearest")
        if quiv is not None:
            s = 2; Xg, Zg = np.meshgrid(xc[::s], zc[::s], indexing="ij")
            qa, qb, mm = quiv
            k = mm[::s, ::s]
            axh.quiver(Xg[k], Zg[k], qa[::s, ::s][k], qb[::s, ::s][k], scale=25, width=0.0022, color="k", alpha=0.55)
        axh.set_title(title, fontsize=11)
        axh.set_xlabel("X [mm]  (Modellkoordinaten)"); axh.set_ylabel("Z [mm]  (Einlasswand unten)")
        return im
    im = plan(ax[0, 0], wr, f"Filteranströmung (Mitte Faltenpaket), relativ zum Mittelwert\n"
              f"γ = {gamma:.3f} (ohne Randzellen {gamma_c:.3f})   Kernfläche unter 50 %: {low*100:.1f} %   über 150 %: {high*100:.1f} %",
              "RdBu_r", norm=TwoSlopeNorm(1.0, 0.0, 2.5))
    plt.colorbar(im, ax=ax[0, 0], fraction=0.035, label="w / w_mittel")
    mj = fluid[:, jc, :]
    im = plan(ax[0, 1], np.where(mj, U[:, jc, :], np.nan),
              f"Geschwindigkeit auf Einlass-Mittelhöhe (Tiefe {dc[jc]:.0f} mm)", "magma", vmax=1.2,
              quiv=(ux[:, jc, :] / u0, uz[:, jc, :] / u0, mj))
    plt.colorbar(im, ax=ax[0, 1], fraction=0.035, label="|u| / u_Einlass")
    ml = fluid[:, jlow, :]
    im = plan(ax[1, 0], np.where(ml, U[:, jlow, :], np.nan),
              f"Geschwindigkeit 12 mm unter den Falten (Tiefe {dc[jlow]:.0f} mm)", "magma", vmax=0.8,
              quiv=(ux[:, jlow, :] / u0, uz[:, jlow, :] / u0, ml))
    plt.colorbar(im, ax=ax[1, 0], fraction=0.035, label="|u| / u_Einlass")
    ms = fluid[ic, :, :]
    im = ax[1, 1].imshow(np.where(ms, U[ic], np.nan), origin="upper", extent=[zc[0], zc[-1], dc[-1], dc[0]],
                         cmap="magma", vmin=0, vmax=1.2, aspect="equal", interpolation="nearest")
    s = 2; Zg, Dg = np.meshgrid(zc[::s], dc[::s]); mm = ms[::s, ::s]
    ax[1, 1].quiver(Zg[mm], Dg[mm], (uz[ic] / u0)[::s, ::s][mm], -(uy[ic] / u0)[::s, ::s][mm],
                    scale=25, width=0.0022, color="w", alpha=0.6)
    ax[1, 1].axhspan(0, a.pleat, color="#c9a55a", alpha=0.18)
    ax[1, 1].set_title(f"Längsschnitt bei X = {xc[ic]:.0f} (Einlassmitte, Faltenzone gelb)", fontsize=11)
    ax[1, 1].set_xlabel("Z [mm]"); ax[1, 1].set_ylabel("Tiefe ab Dichtebene [mm]")
    plt.colorbar(im, ax=ax[1, 1], fraction=0.035, label="|u| / u_Einlass")
    fig.suptitle(f"Strömungsanalyse Rohluftseite – {a.stl.split('/')[-1]}   (Filterwiderstand K = {a.K}, dx = {a.dx} mm)",
                 fontsize=13)
    plt.tight_layout(); plt.savefig(f"airbox_ergebnis{tag}.png", dpi=110)

    res = {"gamma": round(float(gamma), 4), "gamma_ohne_randzellen": round(float(gamma_c), 4),
           "anteil_unter_50pct_kern": round(low, 4), "anteil_ueber_150pct_kern": round(high, 4),
           "anteil_rueckstroemung": round(rev, 4), "max_zu_mittel": round(float(np.nanmax(wr)), 3),
           "zeta_rohluftseite": round(float(zeta), 3),
           "dp_rohluft_mbar_bei_30ms": round(float(zeta) * 0.5 * 1.2 * 30**2 / 100, 2),
           "K": a.K, "dx": a.dx, "steps": a.steps, "verlauf": hist}
    json.dump(res, open(f"airbox_ergebnis{tag}.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "verlauf"}, indent=1))

if __name__ == "__main__":
    main()
