"""Generates the code-architecture diagram (diagram-design skill, Architecture type, full editorial)."""
import sys

PAPER, PAPER2 = "#f6f4ef", "#eef2f7"
INK, MUTED, SOFT = "#1b2f4b", "#3d5a80", "#566880"
ACC, ACC_T, LINK = "#1f6f66", "rgba(42,157,143,0.10)", "#2e5aa8"
SANS, MONO, SERIF = "'Geist', system-ui, sans-serif", "'Geist Mono', ui-monospace, monospace", "'Instrument Serif', serif"


def ink(a):
    return f"rgba(27,47,75,{a})"


TREAT = {  # fill, stroke, tag stroke, tag text
    "backend": ("#ffffff", INK, ink(0.40), INK),
    "focal": (ACC_T, ACC, "rgba(31,111,102,0.50)", ACC),
    "store": (ink(0.05), MUTED, "rgba(61,90,128,0.50)", MUTED),
    "external": (ink(0.03), ink(0.30), ink(0.22), SOFT),
    "optional": (ink(0.02), ink(0.30), ink(0.22), SOFT),
}

W, H = 200, 80
NODES = {  # key: (x, y, treatment, tag, name, sub, sub2)
    "physics": (80, 160, "backend", "LIB", "physics.py", "Drude · band gap · mobility", "376 lines"),
    "electrodes": (80, 304, "backend", "INPUT", "electrodes.py", "9 films: n and μ inputs", "136 lines"),
    "tmm": (80, 448, "backend", "OPTICS", "tmm.py", "thin-film optics (TMM)", "141 lines"),
    "sq": (400, 160, "optional", "CHECK", "sq.py", "Shockley–Queisser check", "43 lines"),
    "device": (400, 304, "focal", "CORE", "device.py", "diode · grid · stripes", "318 lines"),
    "run_all": (720, 160, "backend", "RUN", "run_all.py", "all figures · Monte Carlo", "720 lines"),
    "opaque": (720, 448, "backend", "RUN", "opaque.py", "silver · thick ITO · grid", "166 lines"),
    "data": (1000, 304, "store", "DATA", "data/", "*.csv · numbers.tex", "~60 files"),
    "report": (1000, 448, "external", "PDF", "The report", "pgfplots reads the CSVs", "main.tex → PDF"),
}

# connectors: (path d, colour, marker, width, dashed)
P = []
L = []  # labels: (x, y, w, text, colour)


def arrow(d, kind="muted", dashed=False):
    col, mk, sw = {"muted": (MUTED, "arrow", 1.2), "accent": (ACC, "arrow-accent", 1.4)}[kind]
    P.append((d, col, mk, sw, dashed))


def label(x, y, text, col=SOFT):
    w = 4 * round((len(text) * 5.5 + 8) / 4)
    L.append((x, y, w, text, col))


arrow("M 180,240 V 304")                                   # physics -> electrodes
label(188, 266, "MATERIALS")
arrow("M 280,200 H 400")                                   # physics -> sq
label(312, 180, "SPECTRUM")
arrow("M 280,344 H 400", "accent")                         # electrodes -> device
label(296, 324, "OPTICAL CONST", ACC)
arrow("M 280,488 H 472 Q 480,488 480,480 V 384")  # tmm -> device
label(356, 468, "R, T, A")
arrow("M 600,200 H 720", "muted", True)                    # sq -> run_all
label(640, 180, "CHECK")
arrow("M 600,328 H 772 Q 780,328 780,320 V 240", "accent")  # device -> run_all
label(788, 268, "η PER DESIGN", ACC)
arrow("M 600,360 H 752 Q 760,360 760,368 V 448")           # device -> opaque
label(632, 340, "SAME MODELS")
arrow("M 920,200 H 952 Q 960,200 960,208 V 322 Q 960,330 968,330 H 1000")  # run_all -> data
label(968, 244, "CSV + MACROS")
arrow("M 920,488 H 968 Q 976,488 976,480 V 366 Q 976,358 984,358 H 1000")  # opaque -> data
label(900, 408, "CSV + MACROS")
arrow("M 1100,384 V 448")                                  # data -> report
label(1108, 410, "PGFPLOTS")

ZONES = [  # x, y, w, h, label
    (56, 128, 248, 424, "PHYSICS MODELS"),
    (696, 128, 248, 424, "DRIVER SCRIPTS"),
    (984, 272, 232, 280, "OUTPUTS"),
]


def svg(slug):
    o = []
    a = o.append
    a(f'<svg viewBox="0 88 1280 584" xmlns="http://www.w3.org/2000/svg" role="img" aria-labelledby="{slug}-title {slug}-desc">')
    a(f'<title id="{slug}-title">How the simulation code fits together</title>')
    a(f'<desc id="{slug}-desc">Architecture of the PHYS 3153 transparent-electrode simulation: physics, material and optics '
      f'modules feed the device model, which two driver scripts run to write the data files that the report plots.</desc>')
    a('<defs>')
    a(f'<marker id="arrow" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto"><polygon points="0 0, 8 3, 0 6" fill="{MUTED}"/></marker>')
    a(f'<marker id="arrow-accent" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto"><polygon points="0 0, 8 3, 0 6" fill="{ACC}"/></marker>')
    a(f'<marker id="arrow-link" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto"><polygon points="0 0, 8 3, 0 6" fill="{LINK}"/></marker>')
    a('</defs>')
    a(f'<rect x="0" y="88" width="1280" height="584" fill="{PAPER}"/>')
    # zones
    for x, y, w, h, t in ZONES:
        lw = 4 * round((len(t) * 5.2 + 16) / 4)
        a(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{ink(0.02)}" stroke="{ink(0.10)}" stroke-width="0.8"/>')
        a(f'<rect x="{x + w / 2 - lw / 2:g}" y="{y + 4}" width="{lw}" height="12" rx="2" fill="{PAPER}"/>')
        a(f'<text x="{x + w / 2:g}" y="{y + 13}" fill="{SOFT}" font-size="7" font-family="{MONO}" text-anchor="middle" letter-spacing="0.14em">{t}</text>')
    # connectors
    for d, col, mk, sw, dashed in P:
        da = ' stroke-dasharray="4,3"' if dashed else ""
        a(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="{sw if not dashed else 1}"{da} marker-end="url(#{mk})"/>')
    for x, y, w, t, col in L:
        a(f'<rect x="{x}" y="{y}" width="{w}" height="12" rx="2" fill="{PAPER}"/>')
        a(f'<text x="{x + w / 2:g}" y="{y + 9}" fill="{col}" font-size="8" font-family="{MONO}" text-anchor="middle" letter-spacing="0.06em">{t}</text>')
    # nodes
    for k, (x, y, tr, tag, name, sub, sub2) in NODES.items():
        fill, stroke, tstroke, ttext = TREAT[tr]
        dash = ' stroke-dasharray="4,3"' if tr == "optional" else ""
        tw = 4 * round((len(tag) * 4.9 + 12) / 4)
        cx = x + W / 2
        a(f'<rect x="{x}" y="{y}" width="{W}" height="{H}" rx="6" fill="{PAPER}"/>')
        a(f'<rect x="{x}" y="{y}" width="{W}" height="{H}" rx="6" fill="{fill}" stroke="{stroke}" stroke-width="1"{dash}/>')
        a(f'<rect x="{x + 8}" y="{y + 8}" width="{tw}" height="12" rx="2" fill="transparent" stroke="{tstroke}" stroke-width="0.8"/>')
        a(f'<text x="{x + 8 + tw / 2:g}" y="{y + 17}" fill="{ttext}" font-size="7" font-family="{MONO}" text-anchor="middle" letter-spacing="0.08em">{tag}</text>')
        a(f'<text x="{x + W - 10}" y="{y + 17}" fill="{SOFT}" font-size="8" font-family="{MONO}" text-anchor="end">{sub2}</text>')
        a(f'<text x="{cx:g}" y="{y + 44}" fill="{INK}" font-size="13" font-weight="600" font-family="{SANS}" text-anchor="middle">{name}</text>')
        a(f'<text x="{cx:g}" y="{y + 62}" fill="{MUTED}" font-size="9" font-family="{MONO}" text-anchor="middle">{sub}</text>')
    # legend
    ly = 608
    a(f'<line x1="56" y1="{ly - 8}" x2="1224" y2="{ly - 8}" stroke="{ink(0.10)}" stroke-width="0.8"/>')
    a(f'<text x="56" y="{ly + 10}" fill="{MUTED}" font-size="8" font-family="{MONO}" letter-spacing="0.18em">LEGEND</text>')
    items = [("focal", "The core: scores one electrode in one device"), ("backend", "Python module"),
             ("store", "Generated data"), ("optional", "Sanity check")]
    x = 56
    for tr, t in items:
        fill, stroke, _, _ = TREAT[tr]
        dash = ' stroke-dasharray="3,2"' if tr == "optional" else ""
        a(f'<rect x="{x}" y="{ly + 24}" width="14" height="10" rx="2" fill="{fill}" stroke="{stroke}" stroke-width="1"{dash}/>')
        a(f'<text x="{x + 20}" y="{ly + 32}" fill="{MUTED}" font-size="9" font-family="{SANS}">{t}</text>')
        x += 20 + int(len(t) * 4.7) + 32
    a(f'<line x1="{x}" y1="{ly + 29}" x2="{x + 28}" y2="{ly + 29}" stroke="{ACC}" stroke-width="1.4" marker-end="url(#arrow-accent)"/>')
    a(f'<text x="{x + 36}" y="{ly + 32}" fill="{MUTED}" font-size="9" font-family="{SANS}">Main path of the calculation</text>')
    x += 36 + int(len("Main path of the calculation") * 4.7) + 32
    a(f'<line x1="{x}" y1="{ly + 29}" x2="{x + 28}" y2="{ly + 29}" stroke="{MUTED}" stroke-width="1.2" marker-end="url(#arrow)"/>')
    a(f'<text x="{x + 36}" y="{ly + 32}" fill="{MUTED}" font-size="9" font-family="{SANS}">Uses / feeds</text>')
    a('</svg>')
    return "\n".join(o)


PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Code architecture · Transparent-electrode simulation</title>
  <link href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Geist:wght@400;500;600&family=Geist+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    :root {
      --color-paper: #f6f4ef; --color-paper-2: #eef2f7;
      --color-ink: #1b2f4b; --color-muted: #3d5a80; --color-soft: #566880;
      --color-rule: rgba(27,47,75,0.12);
      --color-accent: #1f6f66; --color-accent-tint: rgba(42,157,143,0.10);
      --font-sans: 'Geist', system-ui, sans-serif;
      --font-serif: 'Instrument Serif', serif;
      --font-mono: 'Geist Mono', ui-monospace, monospace;
    }
    body { font-family: var(--font-sans); background: var(--color-paper); min-height: 100vh; padding: 3rem 2rem; color: var(--color-ink); }
    .container { max-width: 1280px; margin: 0 auto; }
    .header { margin-bottom: 2rem; }
    .header-eyebrow { font-family: var(--font-mono); font-size: 0.66rem; font-weight: 500; letter-spacing: 0.18em; text-transform: uppercase; color: var(--color-muted); margin-bottom: 0.75rem; }
    h1 { font-family: var(--font-serif); font-size: clamp(1.75rem, 3vw + 1rem, 2.6rem); font-weight: 400; letter-spacing: -0.02em; line-height: 1.1; margin-bottom: 0.6rem; }
    .subtitle { font-size: 1rem; line-height: 1.55; color: var(--color-muted); max-width: 70ch; }
    .diagram-container { width: 100%; overflow-x: auto; }
    svg { width: 100%; min-width: 1280px; display: block; }
    .cards { display: grid; grid-template-columns: 1.15fr 1fr 0.9fr; gap: 1rem; margin-top: 1.5rem; }
    @media (max-width: 860px) { .cards { grid-template-columns: 1fr; } }
    .card { background: #fff; border-radius: 6px; border: 1px solid var(--color-rule); padding: 1.25rem; }
    .card .eyebrow { font-family: var(--font-mono); font-size: 0.55rem; letter-spacing: 0.18em; text-transform: uppercase; color: var(--color-muted); margin-bottom: 0.5rem; }
    .card-header { display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.875rem; padding-bottom: 0.875rem; border-bottom: 1px solid rgba(27,47,75,0.08); }
    .card-dot { width: 7px; height: 7px; border-radius: 50%; }
    .card-dot.ink { background: var(--color-ink); } .card-dot.muted { background: var(--color-muted); } .card-dot.coral { background: var(--color-accent); }
    .card h3 { font-size: 0.9rem; font-weight: 600; letter-spacing: -0.005em; }
    .card p, .card ul { color: var(--color-muted); font-size: 0.84rem; line-height: 1.55; list-style: none; }
    .card li { margin-bottom: 0.3rem; padding-left: 0.875rem; position: relative; }
    .card li::before { content: '—'; position: absolute; left: 0; color: rgba(27,47,75,0.25); font-size: 0.75rem; }
    code { font-family: var(--font-mono); font-size: 0.78rem; color: var(--color-ink); }
    .footer { margin-top: 2rem; padding-top: 1.5rem; border-top: 1px solid rgba(27,47,75,0.10); font-family: var(--font-mono); font-size: 0.72rem; letter-spacing: 0.06em; color: var(--color-soft); display: flex; justify-content: space-between; flex-wrap: wrap; gap: 0.5rem; }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <p class="header-eyebrow">Code architecture · PHYS 3153 Project 10</p>
      <h1>How the simulation code fits together</h1>
      <p class="subtitle">Eight Python files, about 1,900 lines with comments. Three physics modules describe the films and the light; <code>device.py</code> turns that into a solar-cell efficiency; two driver scripts run it for every electrode and write the data that every figure and number in the report is drawn from.</p>
    </div>
    <div class="diagram-container">
SVG
    </div>
    <div class="cards">
      <div class="card">
        <p class="eyebrow">Start reading here</p>
        <div class="card-header"><span class="card-dot coral"></span><h3>device.py is the heart</h3></div>
        <p>Given the light that reaches the absorber and the electrode's sheet resistance, it finds the best finger spacing (silicon cell) or stripe width (perovskite module) and returns the efficiency. Everything above it prepares those two inputs; everything below it calls it many times.</p>
      </div>
      <div class="card">
        <p class="eyebrow">The physics modules</p>
        <div class="card-header"><span class="card-dot ink"></span><h3>From two numbers to optics</h3></div>
        <ul>
          <li><code>electrodes.py</code> holds each film's carrier density n and mobility μ</li>
          <li><code>physics.py</code> turns them into a refractive index (Drude + band gap)</li>
          <li><code>tmm.py</code> sends sunlight through the layer stack</li>
        </ul>
      </div>
      <div class="card">
        <p class="eyebrow">Reproducible</p>
        <div class="card-header"><span class="card-dot muted"></span><h3>One command rebuilds it all</h3></div>
        <p><code>python run_all.py</code> then <code>python opaque.py</code>. The random seed is fixed (20261105), so every run gives the same numbers. Nothing in the report is typed in by hand.</p>
      </div>
    </div>
    <div class="footer">
      <span>transparent-electrode simulation · code architecture</span>
      <span>Brandon Deonarine · UWI St. Augustine · 2026</span>
    </div>
  </div>
</body>
</html>
"""

if __name__ == "__main__":
    out = sys.argv[1]
    open(out, "w", encoding="utf-8").write(PAGE.replace("SVG", svg("architecture")))
    print("wrote", out)
