# Airbox Flow

Strömungssimulation für Luftfilterkästen – direkt im Browser, auf der Grafikkarte.

STL des Unterteils laden, Filter-Dichtebene und Einlass anklicken, rechnen. Nach Sekunden bis wenigen Minuten zeigt das Werkzeug, wie gleichmäßig der Filter angeströmt wird, wo Strahlen und Totzonen liegen und wie viel Druckverlust die Rohluftseite kostet. Es gibt keinen Server und keine Installation.

Entstanden beim Umbau eines Golf 2 1.8T (AGU) auf den Luftfilterkasten eines Škoda Roomster 1.9 TDI mit neu konstruiertem, 3D-gedrucktem Unterteil.

---

## Inhalt

| Datei | Zweck |
|---|---|
| `airbox_stroemung.html` | Das Werkzeug. Eine einzelne Datei, im Browser öffnen. |
| `src/engine.js` | Rechenkern: STL einlesen, voxelisieren, Lattice-Boltzmann auf Grafikkarte oder Prozessor, Auswertung. |
| `src/ui.html` | Oberfläche: 3D-Ansicht mit Stromlinien und Schnittebenen, Schritte, Ergebnisse. |
| `src/build.py` | Setzt beide Quelldateien zu `airbox_stroemung.html` zusammen: `python src/build.py` |
| `airbox_cfd.py` | Schlanke Kommandozeilen-Version (Python/NumPy) für Stapelläufe. |

---

## Schnellstart

1. `airbox_stroemung.html` in **Chrome oder Edge** öffnen (ab Version 113, wegen WebGPU). three.js wird beim ersten Öffnen aus dem Netz geladen.
2. **Modell laden:** STL des Unterteils hineinziehen – geschlossener Körper, Einheit mm.
3. **Filter-Dichtebene:** Auf die Auflagefläche der Filterdichtung klicken und die Faltentiefe eintragen (z. B. 40 mm).
4. **Einlass:** Auf die Fläche klicken, in der die Einlassöffnung liegt – die Außenwand neben der Öffnung oder den Rand des Stutzens. Steht das Einlassrohr schräg, auf seine Stirnfläche klicken; das Rechengitter dreht sich dann passend zur Rohrachse. Die erkannte Öffnung wird grün markiert.
5. *Optional:* **Auslass zum LMM** – Lage auf der Dichtfläche, Durchmesser und Höhe des Deckelraums.
6. **Rechnung starten.**

Oben rechts steht, ob eine WebGPU-Grafikkarte erkannt wurde. Ohne Grafikkarte rechnet die Seite auf dem Prozessor, deutlich langsamer. Mit „Filter oben“, den 90°-Knöpfen und „Einpassen“ lässt sich das Modell in der Ansicht drehen; gerechnet wird immer in den Koordinaten der STL.

---

## Was du bekommst

**Kennwerte**

| Wert | Bedeutung | Richtwert |
|---|---|---|
| γ Kern | Gleichförmigkeit der Filteranströmung ohne die wandnahe Randzelle (1 = perfekt) | ab 0,9 gut, ab 0,95 sehr gut |
| Filterfläche unter 50 % | Anteil, der weniger als halb so stark angeströmt wird wie im Mittel (Totzonen) | möglichst unter 10 % |
| Höchste Anströmung | Spitzenwert relativ zum Mittel (Strahl auf das Filtermedium) | möglichst unter 1,5 × |
| ζ Rohluftseite | Druckverlust vom Einlass bis vor den Filter, als Vielfaches des Staudrucks im Einlass | Vergleichswert |

**Strömungsbilder:** Filteranströmung in der Mitte des Faltenpakets, Geschwindigkeit auf Einlasshöhe, Geschwindigkeit knapp unter den Falten und Längsschnitt durch die Einlassmitte.

**Ergebnis in 3D** (in der 3D-Ansicht neben den Schritten):
- **Stromlinien** vom Einlass bis durch den Filter, gefärbt nach Geschwindigkeit. Wo keine Linien hinkommen, wird kaum durchströmt. Auf Wunsch laufen Luftteilchen mit örtlicher Geschwindigkeit die Linien entlang.
- **Schnittebene**, frei verschiebbar parallel zum Filter, längs oder quer zur Einlassrichtung. Zeigt Geschwindigkeit, statischen Druck, **Totaldruckverlust seit dem Einlass** (wo Energie verloren geht: Strahlaufweitung, Umlenkung, Wirbel) oder die Filteranströmung.
- Gehäuse durchsichtig, undurchsichtig oder ausgeblendet; Farbskala mit Werten in der Ansicht.

**Variantentabelle:** Jede Rechnung wird angehängt. Ein Klick auf eine Zeile zeigt deren Bilder wieder.

**Leistungsrechner:** Rechnet ζ für eine PS-Zahl in Luftmasse, Einlassgeschwindigkeit und Druckverlust in mbar um (Benzin-Turbo, rund 0,8 g/s Luft pro PS).

---

## Einstellungen

| Einstellung | Wirkung |
|---|---|
| Zellgröße | 4 mm schnell, 3 mm Standard, 2–2,5 mm fein. Wände müssen mindestens eine Zelle dick sein; sonst meldet die Seite ein Leck. |
| Filter | Papierfilter neu (K = 3), verschmutzt (K = 6), Sportfilter (K = 1,5). |
| Erweitert | Filterwiderstand K, Widerstand quer zu den Falten, Rechenschritte, Viskosität τ, Einlassgeschwindigkeit im Gitter. |

---

## Grenzen

- Gerechnet wird bei **kleinerer Reynoldszahl als im Fahrzeug**, mit Turbulenzmodell. Die Ergebnisse taugen zum **Vergleich von Varianten**, nicht für exakte Druckverluste in mbar.
- Der Filter ist eine poröse Schicht mit einheitlichem Widerstand. Einzelne Falten werden nicht aufgelöst.
- Die Filterebene muss senkrecht zu einer Koordinatenachse liegen. Der Einlass darf in dieser Ebene schräg stehen, aber nicht nach oben oder unten geneigt sein. Gerechnet wird mit einem Einlass.
- Bauteile dünner als etwa eine Zelle werden beim Rechnen aufgedickt oder verschwinden.

---

## Kommandozeilen-Version (`airbox_cfd.py`)

```bash
pip install numpy matplotlib
python airbox_cfd.py Koerper1.stl --tag v1
```

Die Standardwerte passen zu einem Modell, dessen Dichtebene bei Y = 0 liegt (Kasten in +Y) und dessen Einlass auf der Seite mit dem kleinsten Z sitzt. Schräge Einlässe kann nur die Browser-Version.

| Option | Bedeutung | Standard |
|---|---|---|
| `--dx` | Zellgröße in mm | 3.0 |
| `--steps` / `--avg` | Rechenschritte / davon gemittelt | 6000 / 2000 |
| `--depth-axis`, `--depth-sign` | Achse und Richtung der Tiefe ab Dichtebene | `y`, `+` |
| `--seal` | Lage der Dichtebene auf der Tiefenachse (mm) | 0 |
| `--pleat` | Faltentiefe unter der Dichtebene (mm) | 40 |
| `--inlet-face` | Seite des Einlasses (`xmin` … `zmax`) | `zmin` |
| `--K` | Filterwiderstand: Δp Filter / Staudruck im Einlass | 3.0 |
| `--aniso` | Widerstand quer zu den Falten (Faktor) | 10 |
| `--u`, `--tau` | Einlassgeschwindigkeit (Gittereinheiten), Relaxationszeit | 0.06, 0.51 |
| `--tag` | Namenszusatz der Ausgabedateien | – |
| `--from-npz` | nur Auswertung aus einer gespeicherten Rechnung | – |

Ausgabe: `airbox_ergebnis<tag>.png` (Bilder), `airbox_ergebnis<tag>.json` (Kennwerte), `airbox_result<tag>.npz` (Rohdaten).

---

## Technik

- **Löser:** Lattice-Boltzmann D3Q19 im Pull-Schema, Halfway-Bounce-back an Wänden, Zou/He-Einlass, Druckauslass, Smagorinsky-Turbulenzmodell (C_s = 0,17), anisotropes Darcy-Filtermodell. Zeitlich gemittelt.
- **Geometrie:** Parity-Voxelisierung mit 3-facher Überabtastung, automatische Erkennung der Einlassöffnung, gerade Zulaufstrecke vor dem Einlass, Dichtheitsprüfung des Gitters.
- **Grafikkarte:** WebGPU-Compute-Shader (WGSL). Vor jeder Rechnung prüft die Seite die Grafikkarte gegen den Prozessor-Löser.
- **3D-Auswertung:** Felder für Geschwindigkeit, statischen Druck und Totaldruckverlust (bezogen auf den massenstromgewichteten Totaldruck im Einlass) auf dem ganzen Rechengitter; Stromlinien im zeitlich gemittelten Feld mit Runge-Kutta 2. Ordnung und trilinearer Interpolation.
- **Oberfläche:** HTML/JavaScript, Rechnung in einem Web Worker, Darstellung mit three.js r128.
