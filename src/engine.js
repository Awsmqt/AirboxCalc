/* Airbox Flow – Rechenkern: STL einlesen, voxelisieren, Lattice-Boltzmann D3Q19 auf Grafikkarte (WebGPU) oder Prozessor.
   Laeuft im Web Worker der Seite oder in Node (module.exports). */
"use strict";
const CX = [0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0];
const CY = [0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1];
const CZ = [0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1];
const WT = [1 / 3, 1 / 18, 1 / 18, 1 / 18, 1 / 18, 1 / 18, 1 / 18,
  1 / 36, 1 / 36, 1 / 36, 1 / 36, 1 / 36, 1 / 36, 1 / 36, 1 / 36, 1 / 36, 1 / 36, 1 / 36, 1 / 36];
const OPP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17];
const CXf = Float64Array.from(CX), CYf = Float64Array.from(CY), CZf = Float64Array.from(CZ);
const XX = CXf.map((c) => c * c), YY = CYf.map((c) => c * c), ZZ = CZf.map((c) => c * c);
const XY = CXf.map((c, i) => c * CYf[i]), XZ = CXf.map((c, i) => c * CZf[i]), YZ = CYf.map((c, i) => c * CZf[i]);
const firstGE = (arr, v) => { const r = arr.findIndex((d) => d >= v); return r < 0 ? arr.length - 1 : r; };

function parseSTL(buf) {
  const dv = new DataView(buf);
  const n = buf.byteLength >= 84 ? dv.getUint32(80, true) : 0;
  if (n > 0 && 84 + 50 * n === buf.byteLength) {
    const t = new Float32Array(n * 9);
    for (let i = 0; i < n; i++) for (let k = 0; k < 9; k++) t[i * 9 + k] = dv.getFloat32(84 + i * 50 + 12 + k * 4, true);
    return t;
  }
  const txt = new TextDecoder().decode(new Uint8Array(buf));
  const re = /vertex\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)/g; const v = []; let m;
  while ((m = re.exec(txt))) v.push(+m[1], +m[2], +m[3]);
  if (v.length < 9) throw new Error("STL enthält keine Dreiecke.");
  return new Float32Array(v);
}

// Lokales System: X = freie Achse, D = Tiefe ab Dichtebene (in den Kasten), Z = ab Einlassseite nach innen
function makeFrame(p) {
  const dAx = p.sealAxis, iAx = p.inletAxis;
  if (dAx === iAx) throw new Error("Einlassseite und Filterebene liegen auf derselben Achse.");
  const oAx = 3 - dAx - iAx, sd = p.sealSign, si = p.inletSign, seal = p.seal;
  // Schraeger Einlass: lokales Gitter um die Tiefenachse gedreht, damit die Rohrachse auf -Z liegt
  const yaw = (p.inletYaw || 0) * Math.PI / 180, cy = Math.cos(yaw), sy = Math.sin(yaw);
  return {
    dAx, iAx, oAx, sd, si, seal, yaw: p.inletYaw || 0,
    toLocal: (q) => { const a = q[oAx], b = -si * q[iAx]; return [cy * a + sy * b, -sd * (q[dAx] - seal), -sy * a + cy * b]; },
    toModel: (X, D, Z) => { const q = [0, 0, 0]; q[oAx] = cy * X - sy * Z; q[dAx] = seal - sd * D; q[iAx] = -si * (sy * X + cy * Z); return q; },
    dirToModel: (X, D, Z) => { const q = [0, 0, 0]; q[oAx] = cy * X - sy * Z; q[dAx] = -sd * D; q[iAx] = -si * (sy * X + cy * Z); return q; },
  };
}

function voxelize(T, origin, nx, ny, nz, h) {
  const ncol = nx * ny, cnt = new Int32Array(ncol + 1), ntri = T.length / 9;
  const jx = 1.37e-4, jy = 2.71e-4;
  // zwei Durchlaeufe: zaehlen, dann sammeln
  const passes = [null, null]; let hits = null, pos = null;
  for (let pass = 0; pass < 2; pass++) {
    for (let t = 0; t < ntri; t++) {
      const o = t * 9;
      const ax = T[o] - origin[0] - jx, ay = T[o + 1] - origin[1] - jy, az = T[o + 2] - origin[2];
      const bx = T[o + 3] - origin[0] - jx, by = T[o + 4] - origin[1] - jy, bz = T[o + 5] - origin[2];
      const cx = T[o + 6] - origin[0] - jx, cy = T[o + 7] - origin[1] - jy, cz = T[o + 8] - origin[2];
      const den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy);
      if (Math.abs(den) < 1e-12) continue;
      const i0 = Math.max(0, Math.floor(Math.min(ax, bx, cx) / h - 0.5)), i1 = Math.min(nx - 1, Math.ceil(Math.max(ax, bx, cx) / h - 0.5));
      const j0 = Math.max(0, Math.floor(Math.min(ay, by, cy) / h - 0.5)), j1 = Math.min(ny - 1, Math.ceil(Math.max(ay, by, cy) / h - 0.5));
      for (let i = i0; i <= i1; i++) {
        const px = (i + 0.5) * h;
        for (let j = j0; j <= j1; j++) {
          const py = (j + 0.5) * h;
          const l1 = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / den;
          if (l1 < 0) continue;
          const l2 = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / den;
          if (l2 < 0) continue;
          const l3 = 1 - l1 - l2;
          if (l3 < 0) continue;
          const col = i * ny + j;
          if (pass === 0) cnt[col + 1]++;
          else hits[pos[col]++] = l1 * az + l2 * bz + l3 * cz;
        }
      }
    }
    if (pass === 0) {
      for (let c = 0; c < ncol; c++) cnt[c + 1] += cnt[c];
      hits = new Float32Array(cnt[ncol]); pos = Int32Array.from(cnt.subarray(0, ncol));
    }
  }
  const vox = new Uint8Array(nx * ny * nz);
  for (let c = 0; c < ncol; c++) {
    const a = cnt[c], b = cnt[c + 1];
    if (b - a < 2) continue;
    const zs = Array.from(hits.subarray(a, b)).sort((p, q) => p - q);
    const m = zs.length - (zs.length % 2), base = c * nz;
    for (let k = 0; k < m; k += 2) {
      const k0 = Math.max(0, Math.ceil(zs[k] / h - 0.5)), k1 = Math.min(nz - 1, Math.ceil(zs[k + 1] / h - 0.5) - 1);
      for (let kk = k0; kk <= k1; kk++) vox[base + kk] = 1;
    }
  }
  return vox;
}

function components2d(mask, nx, ny) {
  const lab = new Int32Array(nx * ny).fill(-1), comps = []; const q = new Int32Array(nx * ny);
  for (let s = 0; s < nx * ny; s++) {
    if (!mask[s] || lab[s] >= 0) continue;
    const id = comps.length; let qh = 0, qt = 0; q[qt++] = s; lab[s] = id; const cells = [];
    while (qh < qt) {
      const c = q[qh++]; cells.push(c); const i = (c / ny) | 0, j = c % ny;
      const nb = [i > 0 ? c - ny : -1, i < nx - 1 ? c + ny : -1, j > 0 ? c - 1 : -1, j < ny - 1 ? c + 1 : -1];
      for (const d of nb) if (d >= 0 && mask[d] && lab[d] < 0) { lab[d] = id; q[qt++] = d; }
    }
    comps.push(cells);
  }
  return comps;
}

/* Geometrie vorbereiten. p: {tris, dx, sealAxis, sealSign, seal, inletAxis, inletSign, click, pleat} */
function prepare(p) {
  const fr = makeFrame(p), T0 = p.tris, ntri = T0.length / 9;
  const T = new Float32Array(T0.length);
  const mn = [Infinity, Infinity, Infinity], mx = [-Infinity, -Infinity, -Infinity];
  for (let v = 0; v < ntri * 3; v++) {
    const L = fr.toLocal([T0[v * 3], T0[v * 3 + 1], T0[v * 3 + 2]]);
    for (let a = 0; a < 3; a++) { T[v * 3 + a] = L[a]; if (L[a] < mn[a]) mn[a] = L[a]; if (L[a] > mx[a]) mx[a] = L[a]; }
  }
  const h = p.dx, sub = 3, hf = h / sub, duct = 10;
  const out = p.outlet && p.outlet.enabled ? p.outlet : null;
  const origin = [mn[0] - h, (out ? Math.min(mn[1], -out.height) : mn[1]) - h, mn[2] - duct * h];
  const nx = Math.ceil((mx[0] - origin[0] + h) / h), ny = Math.ceil((mx[1] - origin[1] + h) / h), nz = Math.ceil((mx[2] - origin[2] + h) / h);
  const N = nx * ny * nz;
  if (N > 4.5e6) throw new Error(`Gitter zu groß (${(N / 1e6).toFixed(1)} Mio. Zellen) – Zellgröße erhöhen.`);
  const nxf = nx * sub, nyf = ny * sub, nzf = nz * sub;
  const fine = voxelize(T, origin, nxf, nyf, nzf, hf);
  const solid = new Uint8Array(N);
  for (let i = 0; i < nx; i++) for (let j = 0; j < ny; j++) for (let k = 0; k < nz; k++) {
    let s = 0;
    for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) {
      const base = ((i * 3 + a) * nyf + (j * 3 + b)) * nzf + k * 3;
      s += fine[base] + fine[base + 1] + fine[base + 2];
    }
    if (s >= 7) solid[(i * ny + j) * nz + k] = 1;
  }
  const xc = new Float32Array(nx), dc = new Float32Array(ny), zc = new Float32Array(nz);
  for (let i = 0; i < nx; i++) xc[i] = origin[0] + (i + 0.5) * h;
  for (let j = 0; j < ny; j++) dc[j] = origin[1] + (j + 0.5) * h;
  for (let k = 0; k < nz; k++) zc[k] = origin[2] + (k + 0.5) * h;

  // Oeffnungen: Loecher in der vordersten Materialschicht
  const kf = Math.floor((mn[2] + 1.5 - origin[2]) / hf);
  const empty = new Uint8Array(nxf * nyf);
  for (let i = 0; i < nxf; i++) for (let j = 0; j < nyf; j++) empty[i * nyf + j] = fine[(i * nyf + j) * nzf + kf] ? 0 : 1;
  const outside = new Uint8Array(nxf * nyf), q = new Int32Array(nxf * nyf); let qh = 0, qt = 0;
  const push = (c) => { if (empty[c] && !outside[c]) { outside[c] = 1; q[qt++] = c; } };
  for (let i = 0; i < nxf; i++) { push(i * nyf); push(i * nyf + nyf - 1); }
  for (let j = 0; j < nyf; j++) { push(j); push((nxf - 1) * nyf + j); }
  while (qh < qt) {
    const c = q[qh++], i = (c / nyf) | 0, j = c % nyf;
    if (i > 0) push(c - nyf); if (i < nxf - 1) push(c + nyf); if (j > 0) push(c - 1); if (j < nyf - 1) push(c + 1);
  }
  const holes = new Uint8Array(nxf * nyf);
  for (let c = 0; c < holes.length; c++) holes[c] = empty[c] && !outside[c] && (origin[1] + ((c % nyf) + 0.5) * hf > 0) ? 1 : 0;
  const comps = components2d(holes, nxf, nyf).filter((c) => c.length * hf * hf > 25);
  if (!comps.length) throw new Error("Auf der gewählten Einlassseite wurde keine Öffnung gefunden.");
  let best = comps[0];
  if (p.click) {
    const L = fr.toLocal(p.click); let bd = Infinity;
    for (const c of comps) {
      for (let s = 0; s < c.length; s += 3) {
        const x = origin[0] + ((c[s] / nyf | 0) + 0.5) * hf, y = origin[1] + (c[s] % nyf + 0.5) * hf;
        const d = (x - L[0]) ** 2 + (y - L[1]) ** 2; if (d < bd) { bd = d; best = c; }
      }
    }
  } else best = comps.reduce((a, b) => (b.length > a.length ? b : a));
  const sel = new Uint8Array(nx * ny);
  for (const c of best) sel[((c / nyf | 0) / 3 | 0) * ny + ((c % nyf) / 3 | 0)]++;
  const opening = new Uint8Array(nx * ny); let nOpen = 0;
  for (let c = 0; c < nx * ny; c++) if (sel[c] >= 5) { opening[c] = 1; nOpen++; }
  if (nOpen < 8) throw new Error("Einlassöffnung ist zu klein für die gewählte Zellgröße.");

  // Einlassrohr, Boden schliessen, Auslassebene
  const kw0 = zc.findIndex((z) => z >= mn[2]);
  for (let i = 0; i < nx; i++) for (let j = 0; j < ny; j++) for (let k = 0; k < kw0; k++)
    solid[(i * ny + j) * nz + k] = opening[i * ny + j] ? 0 : 1;
  for (let j = 0; j < ny; j++) if (dc[j] > mx[1] - 0.5 * h)
    for (let i = 0; i < nx; i++) for (let k = 0; k < nz; k++) solid[(i * ny + j) * nz + k] = 1;
  const top = dc.findIndex((d) => d >= 0);
  if (top < 0) throw new Error("Dichtebene liegt außerhalb des Modells.");
  const fluid = new Uint8Array(N); const Q = new Int32Array(N); qh = 0; qt = 0;
  const visit = (c, j) => { if (!fluid[c] && !solid[c] && j >= top) { fluid[c] = 1; Q[qt++] = c; } };
  for (let c = 0; c < nx * ny; c++) if (opening[c]) visit(c * nz, c % ny);
  while (qh < qt) {
    const c = Q[qh++], k = c % nz, ij = (c / nz) | 0, j = ij % ny, i = (ij / ny) | 0;
    if (i > 0) visit(c - ny * nz, j); if (i < nx - 1) visit(c + ny * nz, j);
    if (j > 0) visit(c - nz, j - 1); if (j < ny - 1) visit(c + nz, j + 1);
    if (k > 0) visit(c - 1, j); if (k < nz - 1) visit(c + 1, j);
  }
  // Dichtheit: erreicht die Luft den Gitterrand, ist sie durch eine zu duenne Wand nach aussen gelaufen
  let leaked = 0;
  for (let i = 0; i < nx && !leaked; i++) for (let j = top; j < ny && !leaked; j++) for (let k = kw0; k < nz; k++) {
    if ((i === 0 || i === nx - 1 || k === nz - 1) && fluid[(i * ny + j) * nz + k]) { leaked = 1; break; }
  }
  if (leaked) throw new Error(`Bei ${h} mm Zellgröße läuft die Luft durch eine zu dünne Wand nach außen (Wände sollten mindestens so dick wie eine Zelle sein). Bitte eine kleinere Zellgröße wählen.`);
  // Reinluftseite: Deckelraum ueber der Filterflaeche mit Auslass an waehlbarer Stelle
  const outlet = new Uint8Array(N); const outletPts = []; let nOut = 0;
  if (out) {
    const jpt = firstGE(dc, -out.height);
    if (jpt >= top) throw new Error("Deckelhöhe ist kleiner als eine Zelle.");
    const L = fr.toLocal(out.center), r2 = (out.diameter / 2) ** 2;
    for (let i = 0; i < nx; i++) for (let k = 0; k < nz; k++) {
      if (!fluid[(i * ny + top) * nz + k]) continue;
      for (let j = jpt; j < top; j++) { const c = (i * ny + j) * nz + k; fluid[c] = 1; solid[c] = 0; qt++; }
      if ((xc[i] - L[0]) ** 2 + (zc[k] - L[2]) ** 2 <= r2) {
        outlet[(i * ny + jpt) * nz + k] = 1; nOut++;
        const q = fr.toModel(xc[i], dc[jpt], zc[k]); outletPts.push(q[0], q[1], q[2]);
      }
    }
    if (nOut < 4) throw new Error("Der Auslass liegt nicht über der Filterfläche oder ist zu klein.");
  } else {
    for (let i = 0; i < nx; i++) for (let k = 0; k < nz; k++) { const c = (i * ny + top) * nz + k; if (fluid[c]) { outlet[c] = 1; nOut++; } }
  }
  const nf = qt;
  const jm = firstGE(dc, p.pleat / 2);
  let Af = 0; for (let i = 0; i < nx; i++) for (let k = 0; k < nz; k++) Af += fluid[(i * ny + jm) * nz + k];
  // Vorschau-Punkte der Oeffnung (Modellkoordinaten)
  const openPts = [], filterPts = [];
  for (let i = 0; i < nx; i++) for (let k = 0; k < nz; k++) if (fluid[(i * ny + jm) * nz + k]) {
    const q = fr.toModel(xc[i], dc[jm], zc[k]); filterPts.push(q[0], q[1], q[2]);
  }
  for (let c = 0; c < nx * ny; c++) if (opening[c]) {
    const m = fr.toModel(xc[(c / ny) | 0], dc[c % ny], mn[2]); openPts.push(m[0], m[1], m[2]);
  }
  let leak = 0; for (let i = 0; i < nx; i++) for (let k = 0; k < nz; k++) leak += fluid[(i * ny + top) * nz + k];
  return {
    fr, h, nx, ny, nz, N, xc, dc, zc, solid, fluid, opening, outlet, top, jm, nf, mn, mx, pleat: p.pleat,
    summary: {
      grid: [nx, ny, nz], cells: nf, volume_l: nf * h ** 3 / 1e6, inlet_cm2: nOpen * h * h / 100,
      filter_cm2: Af * h * h / 100, outlet_cm2: leak * h * h / 100, nHoles: comps.length, openPts, filterPts, outletPts, outlet_cm2_real: out ? nOut * h * h / 100 : 0,
      pleatOk: jm > top,
    },
  };
}

/* ---------------------------------------------------------------- Lattice-Boltzmann
   p = {K, aniso, u, tau, steps, avg, every}
   Optionale feste Einbauten: ns[a] in 0..1 (partieller Bounce-back, 1 = Wand) */
function setupSim(g, p) {
  const { ny, nz, N, fluid, opening, top, dc } = g;
  const NF = g.nf, cid = new Int32Array(N).fill(-1), cells = new Int32Array(NF);
  let m = 0; for (let c = 0; c < N; c++) if (fluid[c]) { cid[c] = m; cells[m++] = c; }
  const off = CX.map((_, i) => (CX[i] * ny + CY[i]) * nz + CZ[i]);
  const nb = new Int32Array(NF * 19), flag = new Uint32Array(NF);  // 1 Einlass, 2 Auslass, 3 Filter
  const jsOf = new Int32Array(NF);
  let Ain = 0, Lp = 0;
  for (let j = 0; j < g.ny; j++) if (dc[j] > dc[top] && dc[j] <= g.pleat) Lp++;
  for (let a = 0; a < NF; a++) {
    const c = cells[a], k = c % nz, ij = (c / nz) | 0, j = ij % ny;
    jsOf[a] = j;
    for (let i = 0; i < 19; i++) { const s = c - off[i]; nb[a * 19 + i] = s >= 0 && s < N ? cid[s] : -1; }
    if (k === 0 && opening[ij]) { flag[a] = 1; Ain++; }
    else if (g.outlet[c]) flag[a] = 2;
    else if (dc[j] > dc[top] && dc[j] <= g.pleat) flag[a] = 3;
  }
  const layerM = [], layerT = [];
  for (let a = 0; a < NF; a++) { if (jsOf[a] === g.jm) layerM.push(a); if (jsOf[a] === top + 1) layerT.push(a); }
  const Af = layerM.length;
  const u0 = p.u, uface = u0 * Ain / Af, sig = p.K * 0.5 * u0 * u0 / (uface * Math.max(Lp, 1));
  const f0 = new Float32Array(19 * NF);
  for (let i = 0; i < 19; i++) f0.fill(WT[i], i * NF, (i + 1) * NF);
  return { NF, cid, cells, nb, flag, jsOf, Ain, Af, sig, layerM, layerT, f0,
    u0, tau0: p.tau, aniso: p.aniso, cs2s: 18 * Math.SQRT2 * 0.17 * 0.17 };
}

function cpuStep(S, f, f2, acc, ns) {
  const { NF, nb, flag, u0, tau0, sig, aniso, cs2s } = S;
  const fi = S._fi || (S._fi = new Float64Array(19)), fe = S._fe || (S._fe = new Float64Array(19));
  for (let a = 0; a < NF; a++) {
    const base = a * 19;
    let rho = 0, mx = 0, my = 0, mz = 0;
    for (let i = 0; i < 19; i++) {
      const s = nb[base + i];
      const v = s >= 0 ? f[i * NF + s] : f[OPP[i] * NF + a];
      fi[i] = v; rho += v; mx += CX[i] * v; my += CY[i] * v; mz += CZ[i] * v;
    }
    let ux = mx / rho, uy = my / rho, uz = mz / rho;
    const fl = flag[a];
    if (fl === 1 || fl === 2) {
      if (fl === 1) {
        // Dichte aus den Populationen, die aus dem Innenraum kommen (Zou/He-Ansatz)
        let s0 = 0, sm = 0;
        for (let i = 0; i < 19; i++) { if (CZ[i] === 0) s0 += fi[i]; else if (CZ[i] < 0) sm += fi[i]; }
        rho = (s0 + 2 * sm) / (1 - u0); ux = 0; uy = 0; uz = u0;
      } else rho = 1;
      const usq = 1.5 * (ux * ux + uy * uy + uz * uz);
      for (let i = 0; i < 19; i++) {
        const cu = 3 * (CX[i] * ux + CY[i] * uy + CZ[i] * uz);
        f2[i * NF + a] = WT[i] * rho * (1 + cu + 0.5 * cu * cu - usq);
      }
    } else {
      let usq = 1.5 * (ux * ux + uy * uy + uz * uz);
      let pxx = 0, pyy = 0, pzz = 0, pxy = 0, pxz = 0, pyz = 0;
      for (let i = 0; i < 19; i++) {
        const cu = 3 * (CXf[i] * ux + CYf[i] * uy + CZf[i] * uz);
        const e = WT[i] * rho * (1 + cu + 0.5 * cu * cu - usq);
        fe[i] = e;
        const ne = fi[i] - e;
        pxx += XX[i] * ne; pyy += YY[i] * ne; pzz += ZZ[i] * ne;
        pxy += XY[i] * ne; pxz += XZ[i] * ne; pyz += YZ[i] * ne;
      }
      const Qn = Math.sqrt(pxx * pxx + pyy * pyy + pzz * pzz + 2 * (pxy * pxy + pxz * pxz + pyz * pyz));
      const tau = 0.5 * (tau0 + Math.sqrt(tau0 * tau0 + cs2s * Qn / rho));
      if (fl === 3) {
        const sl = 1 - Math.min(tau * sig * aniso, 0.9), sv = 1 - Math.min(tau * sig, 0.9);
        const ex = ux * sl, ey = uy * sv, ez = uz * sl; usq = 1.5 * (ex * ex + ey * ey + ez * ez);
        for (let i = 0; i < 19; i++) {
          const cu = 3 * (CXf[i] * ex + CYf[i] * ey + CZf[i] * ez);
          fe[i] = WT[i] * rho * (1 + cu + 0.5 * cu * cu - usq);
        }
        ux = 0.5 * (ux + ex); uy = 0.5 * (uy + ey); uz = 0.5 * (uz + ez);  // physikalische Geschwindigkeit
      }
      const it = 1 / tau, sv = ns ? ns[a] : 0;
      if (sv > 0) {
        for (let i = 0; i < 19; i++) f2[i * NF + a] = (1 - sv) * (fi[i] - (fi[i] - fe[i]) * it) + sv * fi[OPP[i]];
        ux *= 1 - sv; uy *= 1 - sv; uz *= 1 - sv;
      } else for (let i = 0; i < 19; i++) f2[i * NF + a] = fi[i] - (fi[i] - fe[i]) * it;
    }
    if (acc) { acc[a] += ux; acc[NF + a] += uy; acc[2 * NF + a] += uz; acc[3 * NF + a] += rho; }
  }
}

// vertikale Geschwindigkeit (Richtung Filter) fuer eine Liste von Zellen aus f
function probeCPU(S, f, list) {
  const NF = S.NF, out = new Float32Array(list.length);
  list.forEach((a, n) => { let r = 0, my = 0; for (let i = 0; i < 19; i++) { const v = f[i * NF + a]; r += v; my += CY[i] * v; } out[n] = my / r; });
  return out;
}

function progressInfo(S, vy, step, steps, t0, hist) {
  const nM = S.layerM.length; let sw = 0;
  for (let n = 0; n < nM; n++) sw += -vy[n];
  const wm = sw / nM; let dev = 0; for (let n = 0; n < nM; n++) dev += Math.abs(-vy[n] - wm);
  let qo = 0; for (let n = nM; n < vy.length; n++) qo += -vy[n];
  if (!Number.isFinite(qo)) throw new Error(`Rechnung instabil bei Schritt ${step}. Zellgröße vergrößern oder Viskosität (tau) erhöhen.`);
  const gam = 1 - dev / (2 * nM * wm), mr = qo / (S.Ain * S.u0), el = (Date.now() - t0) / 1000;
  hist.push([step, gam, mr]);
  return { step, steps, gamma: gam, massRatio: mr, elapsed: el, eta: el / step * (steps - step), mlups: S.NF * step / el / 1e6 };
}

function simulate(g, p, onProgress, ns) {
  const S = setupSim(g, p), NF = S.NF;
  let f = S.f0.slice(), f2 = new Float32Array(19 * NF);
  const acc = new Float32Array(4 * NF); let nacc = 0;
  const probeList = S.layerM.concat(S.layerT), t0 = Date.now(), hist = [];
  for (let step = 1; step <= p.steps; step++) {
    const doAcc = step > p.steps - p.avg;
    cpuStep(S, f, f2, doAcc ? acc : null, ns);
    const tmp = f; f = f2; f2 = tmp;
    if (doAcc) nacc++;
    if (step % p.every === 0 || step === p.steps) {
      const info = progressInfo(S, probeCPU(S, f, probeList), step, p.steps, t0, hist);
      info.device = "CPU"; onProgress && onProgress(info);
    }
  }
  for (let x = 0; x < 4 * NF; x++) acc[x] /= Math.max(nacc, 1);
  return evaluate(g, p, acc, S.cells, S.cid, hist, S.Ain, S.Af, ns);
}

/* ---------------------------------------------------------------- WebGPU (mehrere Varianten gleichzeitig) */
const WGSL = /* wgsl */`
struct P { nf: u32, doacc: u32, nprobe: u32, nb: u32, u0: f32, tau0: f32, sig: f32, aniso: f32, cs2s: f32, nlist: u32, pad2: f32, pad3: f32 };
@group(0) @binding(0) var<uniform> prm: P;
@group(0) @binding(1) var<storage, read> fin: array<f32>;
@group(0) @binding(2) var<storage, read_write> fout: array<f32>;
@group(0) @binding(3) var<storage, read> nb: array<i32>;
@group(0) @binding(4) var<storage, read> flg: array<u32>;
@group(0) @binding(5) var<storage, read_write> acc: array<f32>;
@group(0) @binding(8) var<storage, read> ns: array<f32>;

var<private> CX: array<f32, 19> = array<f32, 19>(0., 1., -1., 0., 0., 0., 0., 1., -1., 1., -1., 1., -1., 1., -1., 0., 0., 0., 0.);
var<private> CY: array<f32, 19> = array<f32, 19>(0., 0., 0., 1., -1., 0., 0., 1., -1., -1., 1., 0., 0., 0., 0., 1., -1., 1., -1.);
var<private> CZ: array<f32, 19> = array<f32, 19>(0., 0., 0., 0., 0., 1., -1., 0., 0., 0., 0., 1., -1., -1., 1., 1., -1., -1., 1.);
var<private> WT: array<f32, 19> = array<f32, 19>(0.33333334, 0.055555556, 0.055555556, 0.055555556, 0.055555556, 0.055555556, 0.055555556,
  0.027777778, 0.027777778, 0.027777778, 0.027777778, 0.027777778, 0.027777778, 0.027777778, 0.027777778, 0.027777778, 0.027777778, 0.027777778, 0.027777778);
var<private> OPP: array<u32, 19> = array<u32, 19>(0u, 2u, 1u, 4u, 3u, 6u, 5u, 8u, 7u, 10u, 9u, 12u, 11u, 14u, 13u, 16u, 15u, 18u, 17u);

@compute @workgroup_size(128)
fn lbm_step(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
  let gi = gid.x + gid.y * ng.x * 128u;
  let NF = prm.nf;
  if (gi >= NF * prm.nb) { return; }
  let b = gi / NF;
  let a = gi - b * NF;
  let fb = b * 19u * NF;
  var fi: array<f32, 19>;
  var fe: array<f32, 19>;
  var rho = 0.0; var mx = 0.0; var my = 0.0; var mz = 0.0;
  for (var i = 0u; i < 19u; i = i + 1u) {
    let s = nb[a * 19u + i];
    var v: f32;
    if (s >= 0) { v = fin[fb + i * NF + u32(s)]; } else { v = fin[fb + OPP[i] * NF + a]; }
    fi[i] = v; rho = rho + v; mx = mx + CX[i] * v; my = my + CY[i] * v; mz = mz + CZ[i] * v;
  }
  var ux = mx / rho; var uy = my / rho; var uz = mz / rho;
  let fl = flg[a];
  if (fl == 1u || fl == 2u) {
    if (fl == 1u) {
      var s0 = 0.0; var sm = 0.0;
      for (var i = 0u; i < 19u; i = i + 1u) {
        if (CZ[i] == 0.0) { s0 = s0 + fi[i]; } else if (CZ[i] < 0.0) { sm = sm + fi[i]; }
      }
      rho = (s0 + 2.0 * sm) / (1.0 - prm.u0); ux = 0.0; uy = 0.0; uz = prm.u0;
    } else { rho = 1.0; }
    let usq = 1.5 * (ux * ux + uy * uy + uz * uz);
    for (var i = 0u; i < 19u; i = i + 1u) {
      let cu = 3.0 * (CX[i] * ux + CY[i] * uy + CZ[i] * uz);
      fout[fb + i * NF + a] = WT[i] * rho * (1.0 + cu + 0.5 * cu * cu - usq);
    }
  } else {
    var usq = 1.5 * (ux * ux + uy * uy + uz * uz);
    var pxx = 0.0; var pyy = 0.0; var pzz = 0.0; var pxy = 0.0; var pxz = 0.0; var pyz = 0.0;
    for (var i = 0u; i < 19u; i = i + 1u) {
      let cu = 3.0 * (CX[i] * ux + CY[i] * uy + CZ[i] * uz);
      let e = WT[i] * rho * (1.0 + cu + 0.5 * cu * cu - usq);
      fe[i] = e;
      let ne = fi[i] - e;
      pxx = pxx + CX[i] * CX[i] * ne; pyy = pyy + CY[i] * CY[i] * ne; pzz = pzz + CZ[i] * CZ[i] * ne;
      pxy = pxy + CX[i] * CY[i] * ne; pxz = pxz + CX[i] * CZ[i] * ne; pyz = pyz + CY[i] * CZ[i] * ne;
    }
    let qn = sqrt(pxx * pxx + pyy * pyy + pzz * pzz + 2.0 * (pxy * pxy + pxz * pxz + pyz * pyz));
    let tau = 0.5 * (prm.tau0 + sqrt(prm.tau0 * prm.tau0 + prm.cs2s * qn / rho));
    if (fl == 3u) {
      let sl = 1.0 - min(tau * prm.sig * prm.aniso, 0.9);
      let sv = 1.0 - min(tau * prm.sig, 0.9);
      let ex = ux * sl; let ey = uy * sv; let ez = uz * sl;
      usq = 1.5 * (ex * ex + ey * ey + ez * ez);
      for (var i = 0u; i < 19u; i = i + 1u) {
        let cu = 3.0 * (CX[i] * ex + CY[i] * ey + CZ[i] * ez);
        fe[i] = WT[i] * rho * (1.0 + cu + 0.5 * cu * cu - usq);
      }
      ux = 0.5 * (ux + ex); uy = 0.5 * (uy + ey); uz = 0.5 * (uz + ez);
    }
    let it = 1.0 / tau;
    let s = ns[b * NF + a];
    if (s > 0.0) {
      for (var i = 0u; i < 19u; i = i + 1u) {
        fout[fb + i * NF + a] = (1.0 - s) * (fi[i] - (fi[i] - fe[i]) * it) + s * fi[OPP[i]];
      }
      ux = ux * (1.0 - s); uy = uy * (1.0 - s); uz = uz * (1.0 - s);
    } else {
      for (var i = 0u; i < 19u; i = i + 1u) { fout[fb + i * NF + a] = fi[i] - (fi[i] - fe[i]) * it; }
    }
  }
  if (prm.doacc == 1u) {
    let ab = b * 4u * NF;
    acc[ab + a] = acc[ab + a] + ux; acc[ab + NF + a] = acc[ab + NF + a] + uy;
    acc[ab + 2u * NF + a] = acc[ab + 2u * NF + a] + uz; acc[ab + 3u * NF + a] = acc[ab + 3u * NF + a] + rho;
  }
}

@group(0) @binding(6) var<storage, read> plist: array<u32>;
@group(0) @binding(7) var<storage, read_write> pout: array<f32>;
@compute @workgroup_size(64)
fn probe(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = gid.x;
  if (n >= prm.nprobe) { return; }
  let a = plist[n]; let NF = prm.nf;
  var r = 0.0; var my = 0.0;
  for (var i = 0u; i < 19u; i = i + 1u) { let v = fin[i * NF + a]; r = r + v; my = my + CY[i] * v; }
  pout[n] = my / r;
}

// Gemittelte Werte aus acc fuer eine Liste (Zelle, Komponente) und alle Varianten einsammeln
@group(0) @binding(9) var<storage, read> glist: array<u32>;
@group(0) @binding(10) var<storage, read_write> gout: array<f32>;
@group(0) @binding(11) var<storage, read> accr: array<f32>;
@compute @workgroup_size(64)
fn gather(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = gid.x;
  let total = prm.nlist * prm.nb;
  if (n >= total) { return; }
  let b = n / prm.nlist; let m = n - b * prm.nlist;
  let cell = glist[2u * m]; let comp = glist[2u * m + 1u];
  gout[n] = accr[b * 4u * prm.nf + comp * prm.nf + cell];
}
`;

async function gpuDevice() {
  const gpu = (typeof navigator !== "undefined") && navigator.gpu;
  if (!gpu) return null;
  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) return null;
  const lim = adapter.limits;
  const device = await adapter.requestDevice({ requiredLimits: {
    maxStorageBufferBindingSize: lim.maxStorageBufferBindingSize, maxBufferSize: lim.maxBufferSize,
    maxStorageBuffersPerShaderStage: Math.min(lim.maxStorageBuffersPerShaderStage, 10) } });
  let name = "";
  try { const info = adapter.info || (adapter.requestAdapterInfo ? await adapter.requestAdapterInfo() : {}); name = [info.vendor, info.architecture, info.description].filter(Boolean).join(" "); } catch (e) { }
  return { device, name: name || "Grafikkarte", maxBinding: Math.min(lim.maxStorageBufferBindingSize, lim.maxBufferSize) };
}

class GpuSim {
  /* B Varianten gleichzeitig; ns: Float32Array(B*NF) oder null; glist: Uint32Array [cell,comp,...] fuer gather */
  constructor(dev, S, B = 1, ns = null, glist = null) {
    const d = this.d = dev.device; this.S = S; this.B = B; const NF = S.NF;
    const mk = (size, usage, data) => {
      const b = d.createBuffer({ size: Math.max(16, Math.ceil(size / 4) * 4), usage, mappedAtCreation: !!data });
      if (data) { new data.constructor(b.getMappedRange()).set(data); b.unmap(); }
      return b;
    };
    const ST = GPUBufferUsage.STORAGE, CS = GPUBufferUsage.COPY_SRC, CD = GPUBufferUsage.COPY_DST;
    const finit = new Float32Array(19 * NF * B); for (let b = 0; b < B; b++) finit.set(S.f0, b * 19 * NF);
    this.fA = mk(19 * NF * B * 4, ST | CS | CD, finit); this.fB = mk(19 * NF * B * 4, ST | CS | CD, finit);
    this.nb = mk(NF * 19 * 4, ST, S.nb); this.flg = mk(NF * 4, ST, S.flag);
    this.acc = mk(4 * NF * B * 4, ST | CS | CD);
    this.ns = mk(NF * B * 4, ST | CD, ns || new Float32Array(NF * B));
    const plist = Uint32Array.from(S.layerM.concat(S.layerT)); this.np = plist.length;
    this.plist = mk(plist.length * 4, ST, plist); this.pout = mk(plist.length * 4, ST | CS);
    this.pread = mk(plist.length * 4, GPUBufferUsage.MAP_READ | CD);
    this.nl = glist ? glist.length / 2 : 1;
    this.glist = mk(Math.max(8, this.nl * 8), ST, glist || new Uint32Array(2));
    this.gout = mk(this.nl * B * 4, ST | CS); this.gread = mk(this.nl * B * 4, GPUBufferUsage.MAP_READ | CD);
    const prm = (doacc) => {
      const ab = new ArrayBuffer(48), u = new Uint32Array(ab), fl = new Float32Array(ab);
      u[0] = NF; u[1] = doacc; u[2] = plist.length; u[3] = B; fl[4] = S.u0; fl[5] = S.tau0; fl[6] = S.sig; fl[7] = S.aniso; fl[8] = S.cs2s; u[9] = this.nl;
      const b = d.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | CD }); d.queue.writeBuffer(b, 0, ab); return b;
    };
    this.p0 = prm(0); this.p1 = prm(1);
    const mod = d.createShaderModule({ code: WGSL });
    this.modPromise = mod.getCompilationInfo ? mod.getCompilationInfo() : Promise.resolve({ messages: [] });
    const E = (b, type) => ({ binding: b, visibility: GPUShaderStage.COMPUTE, buffer: { type } });
    const layoutS = d.createBindGroupLayout({ entries: [E(0, "uniform"), E(1, "read-only-storage"), E(2, "storage"),
      E(3, "read-only-storage"), E(4, "read-only-storage"), E(5, "storage"), E(8, "read-only-storage")] });
    const layoutP = d.createBindGroupLayout({ entries: [E(0, "uniform"), E(1, "read-only-storage"), E(6, "read-only-storage"), E(7, "storage")] });
    const layoutG = d.createBindGroupLayout({ entries: [E(0, "uniform"), E(9, "read-only-storage"), E(10, "storage"), E(11, "read-only-storage")] });
    const pl = (l) => d.createPipelineLayout({ bindGroupLayouts: [l] });
    this.pipeS = d.createComputePipeline({ layout: pl(layoutS), compute: { module: mod, entryPoint: "lbm_step" } });
    this.pipeP = d.createComputePipeline({ layout: pl(layoutP), compute: { module: mod, entryPoint: "probe" } });
    this.pipeG = d.createComputePipeline({ layout: pl(layoutG), compute: { module: mod, entryPoint: "gather" } });
    const bg = (pb, fin, fout) => d.createBindGroup({ layout: layoutS, entries: [
      { binding: 0, resource: { buffer: pb } }, { binding: 1, resource: { buffer: fin } }, { binding: 2, resource: { buffer: fout } },
      { binding: 3, resource: { buffer: this.nb } }, { binding: 4, resource: { buffer: this.flg } }, { binding: 5, resource: { buffer: this.acc } },
      { binding: 8, resource: { buffer: this.ns } }] });
    this.bg = { 0: [bg(this.p0, this.fA, this.fB), bg(this.p0, this.fB, this.fA)], 1: [bg(this.p1, this.fA, this.fB), bg(this.p1, this.fB, this.fA)] };
    const bgp = (fin) => d.createBindGroup({ layout: layoutP, entries: [
      { binding: 0, resource: { buffer: this.p0 } }, { binding: 1, resource: { buffer: fin } },
      { binding: 6, resource: { buffer: this.plist } }, { binding: 7, resource: { buffer: this.pout } }] });
    this.bgp = [bgp(this.fA), bgp(this.fB)];
    this.bgg = d.createBindGroup({ layout: layoutG, entries: [
      { binding: 0, resource: { buffer: this.p0 } }, { binding: 9, resource: { buffer: this.glist } },
      { binding: 10, resource: { buffer: this.gout } }, { binding: 11, resource: { buffer: this.acc } }] });
    const groups = Math.ceil(NF * B / 128); this.gx = Math.min(groups, 65535); this.gy = Math.ceil(groups / this.gx);
    this.cur = 0;  // 0: aktuelle Daten in fA
  }
  async check() {
    const info = await this.modPromise;
    const errs = (info.messages || []).filter((m) => m.type === "error");
    if (errs.length) throw new Error("Shader-Fehler: " + errs.map((m) => m.message).join("; "));
  }
  curBuf() { return this.cur === 0 ? this.fA : this.fB; }
  steps(n, doacc) {
    const enc = this.d.createCommandEncoder(), pass = enc.beginComputePass();
    pass.setPipeline(this.pipeS);
    for (let s = 0; s < n; s++) { pass.setBindGroup(0, this.bg[doacc][this.cur]); pass.dispatchWorkgroups(this.gx, this.gy); this.cur ^= 1; }
    pass.end(); this.d.queue.submit([enc.finish()]);
  }
  setNS(ns) { this.d.queue.writeBuffer(this.ns, 0, ns); }
  clearAcc() { const enc = this.d.createCommandEncoder(); enc.clearBuffer(this.acc); this.d.queue.submit([enc.finish()]); }
  // Startzustand aller Varianten auf eine gespeicherte Verteilung (19*NF) setzen
  resetFrom(base) {
    const enc = this.d.createCommandEncoder(), sz = 19 * this.S.NF * 4;
    for (let b = 0; b < this.B; b++) enc.copyBufferToBuffer(base, 0, this.curBuf(), b * sz, sz);
    this.d.queue.submit([enc.finish()]);
  }
  async probe() {
    const enc = this.d.createCommandEncoder(), pass = enc.beginComputePass();
    pass.setPipeline(this.pipeP); pass.setBindGroup(0, this.bgp[this.cur]); pass.dispatchWorkgroups(Math.ceil(this.np / 64)); pass.end();
    enc.copyBufferToBuffer(this.pout, 0, this.pread, 0, this.np * 4); this.d.queue.submit([enc.finish()]);
    await this.pread.mapAsync(GPUMapMode.READ); const out = new Float32Array(this.pread.getMappedRange()).slice(); this.pread.unmap();
    return out;
  }
  async gather() {
    const enc = this.d.createCommandEncoder(), pass = enc.beginComputePass();
    pass.setPipeline(this.pipeG); pass.setBindGroup(0, this.bgg); pass.dispatchWorkgroups(Math.ceil(this.nl * this.B / 64)); pass.end();
    enc.copyBufferToBuffer(this.gout, 0, this.gread, 0, this.nl * this.B * 4); this.d.queue.submit([enc.finish()]);
    await this.gread.mapAsync(GPUMapMode.READ); const out = new Float32Array(this.gread.getMappedRange()).slice(); this.gread.unmap();
    return out;
  }
  async read(buf, n, offset = 0) {
    const rb = this.d.createBuffer({ size: n * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = this.d.createCommandEncoder(); enc.copyBufferToBuffer(buf, offset * 4, rb, 0, n * 4); this.d.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ); const out = new Float32Array(rb.getMappedRange()).slice(); rb.unmap(); rb.destroy();
    return out;
  }
  // aktuellen Zustand von Variante b in einen eigenen Puffer kopieren
  snapshot(b = 0) {
    const sz = 19 * this.S.NF * 4;
    const buf = this.d.createBuffer({ size: sz, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const enc = this.d.createCommandEncoder(); enc.copyBufferToBuffer(this.curBuf(), b * sz, buf, 0, sz); this.d.queue.submit([enc.finish()]);
    return buf;
  }
  destroy() { for (const b of [this.fA, this.fB, this.nb, this.flg, this.acc, this.ns, this.plist, this.pout, this.pread, this.glist, this.gout, this.gread, this.p0, this.p1]) b.destroy(); }
}

// GPU gegen CPU pruefen: 2 Varianten (ohne und mit Einbauten), 12 Schritte vom gleichen Startzustand
async function gpuSelfTest(dev, g, p) {
  const S = setupSim(g, p), NF = S.NF, n = 12;
  const ns = new Float32Array(2 * NF);
  for (let a = 0; a < NF; a += 17) if (S.flag[a] === 0) ns[NF + a] = 1;
  const sim = new GpuSim(dev, S, 2, ns); await sim.check();
  sim.steps(n, 0);
  const fg = await sim.read(sim.curBuf(), 19 * NF * 2); sim.destroy();
  let md = 0;
  for (let b = 0; b < 2; b++) {
    let f = S.f0.slice(), f2 = new Float32Array(19 * NF); const nsb = ns.subarray(b * NF, (b + 1) * NF);
    for (let s = 0; s < n; s++) { cpuStep(S, f, f2, null, nsb); const t = f; f = f2; f2 = t; }
    for (let x = 0; x < f.length; x += 7) md = Math.max(md, Math.abs(f[x] - fg[b * 19 * NF + x]));
  }
  return md;
}

async function simulateGPU(dev, g, p, onProgress, ns) {
  const S = setupSim(g, p), NF = S.NF, sim = new GpuSim(dev, S, 1, ns); await sim.check();
  const t0 = Date.now(), hist = []; let step = 0, nacc = 0;
  const startAcc = p.steps - p.avg;
  while (step < p.steps) {
    let n = Math.min(p.every, p.steps - step);
    if (step < startAcc && step + n > startAcc) n = startAcc - step;
    const doacc = step >= startAcc ? 1 : 0;
    sim.steps(n, doacc); step += n; if (doacc) nacc += n;
    if (step % p.every === 0 || step === p.steps) {
      const info = progressInfo(S, await sim.probe(), step, p.steps, t0, hist);
      info.device = "GPU"; onProgress && onProgress(info);
    }
  }
  const acc = await sim.read(sim.acc, 4 * NF); sim.destroy();
  for (let x = 0; x < 4 * NF; x++) acc[x] /= Math.max(nacc, 1);
  return evaluate(g, p, acc, S.cells, S.cid, hist, S.Ain, S.Af, ns);
}

function evaluate(g, p, acc, cells, cid, hist, Ain, Af, ns) {
  const { nx, ny, nz, xc, dc, zc, opening, top, h } = g, NF = g.nf, u0 = p.u;
  const fluid = ns ? g.fluid.map((v, c) => (v && !(ns[cid[c]] > 0.5) ? 1 : 0)) : g.fluid;
  const U = (c, comp) => { const a = cid[c]; return a < 0 ? NaN : acc[comp * NF + a]; };
  const idx = (i, j, k) => (i * ny + j) * nz + k;
  // Filterkarte (Mitte Faltenpaket)
  const jm = g.jm, fmap = new Float32Array(nx * nz).fill(NaN);
  let sw = 0, n = 0;
  for (let i = 0; i < nx; i++) for (let k = 0; k < nz; k++) { const c = idx(i, jm, k); if (fluid[c]) { const w = -U(c, 1); fmap[i * nz + k] = w; sw += w; n++; } }
  const wm = sw / n; let dev = 0, cdev = 0, cn = 0, csum = 0, low = 0, high = 0, rev = 0, mxr = 0;
  const core = new Uint8Array(nx * nz);
  for (let i = 1; i < nx - 1; i++) for (let k = 1; k < nz - 1; k++) {
    const ok = (ii, kk) => fluid[idx(ii, jm, kk)];
    if (ok(i, k) && ok(i - 1, k) && ok(i + 1, k) && ok(i, k - 1) && ok(i, k + 1)) core[i * nz + k] = 1;
  }
  for (let x = 0; x < nx * nz; x++) if (core[x]) { csum += fmap[x]; cn++; }
  const cm = csum / cn;
  for (let x = 0; x < nx * nz; x++) {
    const w = fmap[x]; if (Number.isNaN(w)) continue;
    dev += Math.abs(w - wm); if (w < 0) rev++; mxr = Math.max(mxr, w / wm);
    if (core[x]) { cdev += Math.abs(w - cm); if (w < 0.5 * cm) low++; if (w > 1.5 * cm) high++; }
    fmap[x] = w / wm;
  }
  // Druckverlust Rohluftseite
  let pin = 0, pinN = 0, pb = 0, pbN = 0;
  const jb = firstGE(dc, g.pleat + 2 * h);
  for (let i = 0; i < nx; i++) for (let j = 0; j < ny; j++) if (opening[i * ny + j]) { const c = idx(i, j, 1); if (fluid[c]) { pin += U(c, 3); pinN++; } }
  for (let i = 0; i < nx; i++) for (let k = 0; k < nz; k++) { const c = idx(i, jb, k); if (fluid[c]) { pb += U(c, 3); pbN++; } }
  const zeta = ((pin / pinN - pb / pbN) / 3) / (0.5 * u0 * u0);
  // Schnitte
  let si = 0, sj = 0, so = 0;
  for (let i = 0; i < nx; i++) for (let j = 0; j < ny; j++) if (opening[i * ny + j]) { si += i; sj += j; so++; }
  const ic = Math.round(si / so), jc = Math.round(sj / so);
  const jlow = firstGE(dc, g.pleat + 12);
  const plan = (j) => {
    const s = new Float32Array(nx * nz).fill(NaN), a = new Float32Array(nx * nz), b = new Float32Array(nx * nz);
    for (let i = 0; i < nx; i++) for (let k = 0; k < nz; k++) {
      const c = idx(i, j, k); if (!fluid[c]) continue;
      const ux = U(c, 0) / u0, uy = U(c, 1) / u0, uz = U(c, 2) / u0;
      s[i * nz + k] = Math.hypot(ux, uy, uz); a[i * nz + k] = ux; b[i * nz + k] = uz;
    }
    return { speed: s, ua: a, ub: b, n1: nx, n2: nz, depth: dc[j] };
  };
  const lon = { speed: new Float32Array(nz * ny).fill(NaN), ua: new Float32Array(nz * ny), ub: new Float32Array(nz * ny), n1: nz, n2: ny, x: xc[ic] };
  for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) {
    const c = idx(ic, j, k); if (!fluid[c]) continue;
    const ux = U(c, 0) / u0, uy = U(c, 1) / u0, uz = U(c, 2) / u0;
    lon.speed[k * ny + j] = Math.hypot(ux, uy, uz); lon.ua[k * ny + j] = uz; lon.ub[k * ny + j] = uy;
  }
  return {
    metrics: {
      gamma: 1 - dev / (2 * n * wm), gammaCore: 1 - cdev / (2 * cn * cm), lowCore: low / cn, highCore: high / cn,
      reverse: rev / n, maxToMean: mxr, zeta, inlet_cm2: Ain * h * h / 100, filter_cm2: Af * h * h / 100,
    },
    hist, fmap, nx, nz, ny, xc: Array.from(xc), zc: Array.from(zc), dc: Array.from(dc), pleat: g.pleat,
    planInlet: plan(jc), planLow: plan(jlow), lon,
    frame: { dAx: g.fr.dAx, iAx: g.fr.iAx, oAx: g.fr.oAx, sd: g.fr.sd, si: g.fr.si, seal: g.fr.seal, yaw: g.fr.yaw },
    vol: field3d(g, p, acc, cid, fluid, pb / pbN), lines: streamlines(g, p, acc, cid, fluid),
  };
}

/* ---------------------------------------------------------------- 3D-Auswertung
   Felder auf dem ganzen Rechengitter (NaN = keine Luft), bezogen auf den Staudruck im Einlass q = rho/2 u0^2:
   spd  = |u| / u0
   pst  = statischer Druck relativ zum Raum unter dem Filter
   loss = Totaldruckverlust seit dem Einlass (wo geht Energie verloren?) */
function field3d(g, p, acc, cid, fluid, rhoRef) {
  const { nx, ny, nz, opening, h } = g, NF = g.nf, u0 = p.u, q = 0.5 * u0 * u0, N = nx * ny * nz;
  const spd = new Float32Array(N).fill(NaN), pst = new Float32Array(N).fill(NaN), loss = new Float32Array(N).fill(NaN);
  const ptot = (a) => { const ux = acc[a], uy = acc[NF + a], uz = acc[2 * NF + a]; return acc[3 * NF + a] / 3 + 0.5 * (ux * ux + uy * uy + uz * uz); };
  // Bezug: massenstromgewichteter Totaldruck im Einlassquerschnitt, zwei Zellen vor dem Kasten.
  // Die kuenstliche Zulaufstrecke davor wird nicht dargestellt.
  const kB = Math.max(2, g.zc.findIndex((z) => z >= g.mn[2]) - 2);
  let pin = 0, win = 0;
  for (let i = 0; i < nx; i++) for (let j = 0; j < ny; j++) if (opening[i * ny + j]) {
    const c = (i * ny + j) * nz + kB, a = cid[c]; if (a < 0 || !fluid[c]) continue;
    const w = Math.max(acc[2 * NF + a], 0); pin += w * ptot(a); win += w;
  }
  pin /= Math.max(win, 1e-12);
  for (let c = 0; c < N; c++) {
    const a = cid[c]; if (a < 0 || !fluid[c] || c % nz < kB) continue;
    const ux = acc[a], uy = acc[NF + a], uz = acc[2 * NF + a];
    spd[c] = Math.hypot(ux, uy, uz) / u0;
    pst[c] = (acc[3 * NF + a] - rhoRef) / 3 / q;
    loss[c] = (pin - ptot(a)) / q;
  }
  return { nx, ny, nz, h, x0: g.xc[0], d0: g.dc[0], z0: g.zc[0], spd, pst, loss, jm: g.jm, top: g.top, pleat: g.pleat, kB, zIn: g.mn[2] };
}
// Mittlere Stromlinien vom Einlass aus (zeitlich gemitteltes Feld, Runge-Kutta 2. Ordnung, Schritt 0,35 Zellen)
function streamlines(g, p, acc, cid, fluid, maxSeeds = 220) {
  const { nx, ny, nz, opening, h, xc, dc, zc, top, fr } = g, NF = g.nf, u0 = p.u;
  const vel = (x, y, z, out) => {  // trilinear, Wandzellen = 0
    const i0 = Math.floor(x), j0 = Math.floor(y), k0 = Math.floor(z), fx = x - i0, fy = y - j0, fz = z - k0;
    out[0] = out[1] = out[2] = 0; let wsum = 0;
    for (let a = 0; a < 2; a++) for (let b = 0; b < 2; b++) for (let c = 0; c < 2; c++) {
      const i = i0 + a, j = j0 + b, k = k0 + c; if (i < 0 || j < 0 || k < 0 || i >= nx || j >= ny || k >= nz) continue;
      const w = (a ? fx : 1 - fx) * (b ? fy : 1 - fy) * (c ? fz : 1 - fz), cc = (i * ny + j) * nz + k, q = cid[cc];
      if (q < 0 || !fluid[cc]) continue;
      out[0] += w * acc[q]; out[1] += w * acc[NF + q]; out[2] += w * acc[2 * NF + q]; wsum += w;
    }
    return wsum;
  };
  const inside = (x, y, z) => { const i = Math.round(x), j = Math.round(y), k = Math.round(z); if (i < 0 || j < 0 || k < 0 || i >= nx || j >= ny || k >= nz) return false; return !!fluid[(i * ny + j) * nz + k]; };
  // Startpunkte gleichmaessig ueber die Einlassoeffnung, kurz vor dem Kasten
  const seeds = [], kS = Math.max(2, (zc.findIndex((z) => z >= g.mn[2]) | 0) - 3);
  for (let i = 0; i < nx; i++) for (let j = 0; j < ny; j++) if (opening[i * ny + j]) for (const [dx, dy] of [[-0.25, -0.25], [0.25, -0.25], [-0.25, 0.25], [0.25, 0.25]]) seeds.push([i + dx, j + dy, kS]);
  const stride = Math.max(1, seeds.length / maxSeeds);
  const pts = [], spd = [], off = [0], v1 = [0, 0, 0], v2 = [0, 0, 0], ds = 0.35;
  for (let s = 0; s < seeds.length; s += stride) {
    let [x, y, z] = seeds[Math.floor(s)]; const P = [], V = [];
    for (let st = 0; st < 6000; st++) {
      if (!inside(x, y, z) || vel(x, y, z, v1) < 0.3) break;
      const m1 = Math.hypot(v1[0], v1[1], v1[2]); if (m1 < 0.01 * u0) break;
      if (st % 3 === 0) { P.push(x, y, z); V.push(m1 / u0); }
      const xm = x + 0.5 * ds * v1[0] / m1, ym = y + 0.5 * ds * v1[1] / m1, zm = z + 0.5 * ds * v1[2] / m1;
      if (vel(xm, ym, zm, v2) < 0.3) break;
      const m2 = Math.hypot(v2[0], v2[1], v2[2]); if (m2 < 0.01 * u0) break;
      x += ds * v2[0] / m2; y += ds * v2[1] / m2; z += ds * v2[2] / m2;
      if (y < top + 0.5) { P.push(x, y, z); V.push(m2 / u0); break; }  // durch den Filter auf der Reinluftseite angekommen
    }
    if (P.length < 6) continue;
    for (let q = 0; q < P.length; q += 3) { const m = fr.toModel(xc[0] + P[q] * h, dc[0] + P[q + 1] * h, zc[0] + P[q + 2] * h); pts.push(m[0], m[1], m[2]); }
    for (const v of V) spd.push(v);
    off.push(spd.length);
  }
  return { pts: Float32Array.from(pts), spd: Float32Array.from(spd), off: Uint32Array.from(off) };
}

// ---------------------------------------------------------------- Worker-Anbindung
if (typeof self !== "undefined" && typeof importScripts === "function") {
  let tris = null, geo = null, dev = null, devTried = false;
  const getDev = async () => { if (!devTried) { devTried = true; try { dev = await gpuDevice(); } catch (err) { dev = null; } } return dev; };
  const post = (m, tr) => self.postMessage(m, tr || []);
  async function runSim(p) {
    const prog = (info) => post({ type: "progress", info });
    let res = null;
    if (p.device !== "cpu" && await getDev()) {
      try {
        post({ type: "status", text: "Grafikkarte wird geprüft …" });
        const md = await gpuSelfTest(dev, geo, p);
        if (!(md < 1e-3)) throw new Error("Abweichung GPU/CPU " + md.toExponential(1));
        post({ type: "status", text: "Rechnet auf der Grafikkarte (" + dev.name + ")" });
        res = await simulateGPU(dev, geo, p, prog, null); res.device = "GPU · " + dev.name;
      } catch (err) { post({ type: "status", text: "Grafikkarte nicht nutzbar (" + (err.message || err) + "), rechne auf dem Prozessor" }); res = null; }
    } else if (p.device !== "cpu") post({ type: "status", text: "Keine WebGPU-Grafikkarte verfügbar, rechne auf dem Prozessor" });
    if (!res) { res = simulate(geo, p, prog, null); res.device = "CPU"; }
    return res;
  }
  self.onmessage = async (e) => {
    const m = e.data;
    try {
      if (m.type === "load") {
        tris = parseSTL(m.buffer); geo = null;
        const mn = [Infinity, Infinity, Infinity], mx = [-Infinity, -Infinity, -Infinity];
        for (let v = 0; v < tris.length; v += 3) for (let a = 0; a < 3; a++) { mn[a] = Math.min(mn[a], tris[v + a]); mx[a] = Math.max(mx[a], tris[v + a]); }
        const copy = tris.slice();
        post({ type: "loaded", tris: copy, ntri: tris.length / 9, mn, mx }, [copy.buffer]);
      } else if (m.type === "gpuinfo") {
        await getDev(); post({ type: "gpuinfo", ok: !!dev, name: dev ? dev.name : "" });
      } else if (m.type === "prepare") {
        const t = Date.now();
        geo = prepare(Object.assign({ tris }, m.params));
        post({ type: "prepared", summary: geo.summary, ms: Date.now() - t });
      } else if (m.type === "run") {
        if (!geo) throw new Error("Bitte zuerst die Geometrie prüfen.");
        const res = await runSim(m.params), v = res.vol, L = res.lines;
        post({ type: "result", res }, [v.spd.buffer, v.pst.buffer, v.loss.buffer, L.pts.buffer, L.spd.buffer, L.off.buffer]);
      }
    } catch (err) { post({ type: "error", message: err.message || String(err), during: m.type }); }
  };
}
if (typeof module !== "undefined") module.exports = { parseSTL, makeFrame, prepare, setupSim, cpuStep, simulate, evaluate, field3d, streamlines };
