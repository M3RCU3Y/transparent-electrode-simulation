"""Generates the 'how one electrode is scored' Steps diagram (diagram-design skill)."""
import sys

PAPER = "#f6f4ef"
INK, MUTED, SOFT = "#1b2f4b", "#3d5a80", "#566880"
ACC, ACC_T = "#1f6f66", "rgba(42,157,143,0.10)"
RULE_SOLID = "#c9d2de"
SANS, MONO, SERIF = "'Geist', system-ui, sans-serif", "'Geist Mono', ui-monospace, monospace", "'Instrument Serif', serif"

STEPS = [  # slug, title, two-line description, medallion glyph, file
    ("inputs", "Film inputs", ["carrier density n and", "mobility μ (literature)"], "n, μ", "electrodes.py"),
    ("optics", "Optical constants", ["Drude free electrons", "+ band-gap absorption"], "ε(λ)", "physics.py"),
    ("stack", "Light in the stack", ["transfer matrices give", "R, T and A at every λ"], "R T A", "tmm.py"),
    ("current", "Photocurrent", ["absorbed light × AM1.5G", "sunlight spectrum"], "J", "device.py"),
    ("losses", "Device losses", ["shading + sideways Rs;", "best finger/stripe gap"], "Rs", "device.py"),
    ("eta", "Efficiency", ["diode maximum-power", "point gives η"], "η", "device.py"),
    ("best", "Best design", ["keep the thickness", "with the highest η"], "max", "device.optimise"),
]
FOCAL = "losses"

n = len(STEPS)
SW = min(224, 4 * ((1280 - 160) // (4 * n)))
X0 = (1280 - n * SW) // 2
RAIL_Y, RAIL_H = 240, 48


def cx(i):
    return X0 + i * SW + SW // 2


def svg(slug):
    o = []
    a = o.append
    a(f'<svg viewBox="0 128 1280 432" xmlns="http://www.w3.org/2000/svg" role="img" aria-labelledby="{slug}-title {slug}-desc">')
    a(f'<title id="{slug}-title">How one electrode gets its score</title>')
    a(f'<desc id="{slug}-desc">Seven stages that turn a film\'s carrier density and mobility into a solar-cell efficiency: optical '
      f'constants, thin-film optics, photocurrent, device losses, efficiency, then the best thickness is kept.</desc>')
    a(f'<rect x="0" y="128" width="1280" height="432" fill="{PAPER}"/>')
    # rail
    a('<g aria-hidden="true">')
    for i, (s, *_r) in enumerate(STEPS):
        f = s == FOCAL
        fill = ACC_T if f else ("rgba(27,47,75,0.05)" if i % 2 == 0 else "rgba(27,47,75,0.09)")
        extra = ' data-focal="true"' if f else ""
        a(f'<rect x="{X0 + i * SW}" y="{RAIL_Y}" width="{SW}" height="{RAIL_H}" fill="{fill}" data-role="rail" data-step="{s}" data-index="{i + 1}"{extra}/>')
    fi = [s for s, *_ in STEPS].index(FOCAL)
    a(f'<line x1="{X0 + fi * SW}" y1="{RAIL_Y + 0.5}" x2="{X0 + (fi + 1) * SW}" y2="{RAIL_Y + 0.5}" stroke="{ACC}" stroke-width="1"/>')
    a(f'<line x1="{X0 + fi * SW}" y1="{RAIL_Y + RAIL_H - 0.5}" x2="{X0 + (fi + 1) * SW}" y2="{RAIL_Y + RAIL_H - 0.5}" stroke="{ACC}" stroke-width="1"/>')
    a('</g>')
    # numbers + guides
    a(f'<g font-family="{MONO}" font-size="8" font-weight="600" letter-spacing="0.14em" text-anchor="middle">')
    for i, (s, *_r) in enumerate(STEPS):
        f = s == FOCAL
        a(f'<text x="{cx(i)}" y="164" fill="{ACC if f else MUTED}" data-role="number" data-step="{s}">{i + 1:02d}</text>')
    a('</g>')
    for i, (s, *_r) in enumerate(STEPS):
        a(f'<line x1="{cx(i)}" y1="184" x2="{cx(i)}" y2="220" stroke="{ACC if s == FOCAL else RULE_SOLID}" stroke-width="1" aria-hidden="true"/>')
    # medallions + glyphs
    a(f'<g fill="{PAPER}" stroke-width="1">')
    for i, (s, t, d, g, fl) in enumerate(STEPS):
        f = s == FOCAL
        if f:
            a(f'<circle cx="{cx(i)}" cy="264" r="36" stroke="none"/>')
            a(f'<circle cx="{cx(i)}" cy="264" r="36" fill="{ACC_T}" stroke="{ACC}" data-role="medallion" data-step="{s}" data-focal="true"/>')
        else:
            a(f'<circle cx="{cx(i)}" cy="264" r="36" stroke="{INK}" data-role="medallion" data-step="{s}"/>')
    a('</g>')
    for i, (s, t, d, g, fl) in enumerate(STEPS):
        f = s == FOCAL
        a(f'<text x="{cx(i)}" y="271" fill="{ACC if f else INK}" font-size="21" font-style="italic" font-family="{SERIF}" text-anchor="middle" aria-hidden="true">{g}</text>')
    # copy
    for i, (s, t, d, g, fl) in enumerate(STEPS):
        f = s == FOCAL
        a(f'<text x="{cx(i)}" y="340" fill="{ACC if f else INK}" font-size="13" font-weight="600" font-family="{SANS}" text-anchor="middle">{t}</text>')
        for k, line in enumerate(d):
            a(f'<text x="{cx(i)}" y="{366 + 18 * k}" fill="{MUTED}" font-size="9" font-family="{MONO}" text-anchor="middle">{line}</text>')
        tw = 4 * round((len(fl) * 4.6 + 14) / 4)
        a(f'<rect x="{cx(i) - tw / 2:g}" y="{410}" width="{tw}" height="14" rx="2" fill="transparent" stroke="rgba(27,47,75,0.22)" stroke-width="0.8"/>')
        a(f'<text x="{cx(i)}" y="420" fill="{SOFT}" font-size="7.5" font-family="{MONO}" text-anchor="middle" letter-spacing="0.04em">{fl}</text>')
    # bracket: stages 03-06 are repeated for every thickness, chosen in 07
    bx1, bx2, by = X0 + 2 * SW + 16, X0 + 6 * SW - 16, 452
    a(f'<path d="M {bx1},{by - 8} V {by} H {bx2} V {by - 8}" fill="none" stroke="{SOFT}" stroke-width="1" stroke-dasharray="4,3"/>')
    lab = "REPEATED FOR EVERY THICKNESS, 20–1000 NM"
    lw = 4 * round((len(lab) * 5.3 + 16) / 4)
    mid = (bx1 + bx2) / 2
    a(f'<rect x="{mid - lw / 2:g}" y="{by + 8}" width="{lw}" height="12" rx="2" fill="{PAPER}"/>')
    a(f'<text x="{mid:g}" y="{by + 17}" fill="{SOFT}" font-size="8" font-family="{MONO}" text-anchor="middle" letter-spacing="0.06em">{lab}</text>')
    # legend
    ly = 508
    a(f'<line x1="80" y1="{ly - 8}" x2="1200" y2="{ly - 8}" stroke="rgba(27,47,75,0.10)" stroke-width="0.8"/>')
    a(f'<text x="80" y="{ly + 10}" fill="{MUTED}" font-size="8" font-family="{MONO}" letter-spacing="0.18em">LEGEND</text>')
    a(f'<circle cx="87" cy="{ly + 32}" r="7" fill="{PAPER}" stroke="{INK}" stroke-width="1"/>')
    a(f'<text x="100" y="{ly + 35}" fill="{MUTED}" font-size="9" font-family="{SANS}">Stage</text>')
    a(f'<circle cx="151" cy="{ly + 32}" r="7" fill="{ACC_T}" stroke="{ACC}" stroke-width="1"/>')
    a(f'<text x="164" y="{ly + 35}" fill="{MUTED}" font-size="9" font-family="{SANS}">Where the device design changes the answer</text>')
    a(f'<text x="1200" y="{ly + 35}" fill="{MUTED}" font-size="13" font-style="italic" font-family="{SERIF}" text-anchor="end">'
      f'Then the whole chain runs again for every electrode, both devices, and 300 random-input trials.</text>')
    a('</svg>')
    return "\n".join(o)


PAGE = open(sys.argv[2], encoding="utf-8").read() if len(sys.argv) > 2 else None

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Scoring pipeline · Transparent-electrode simulation</title>
  <link href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Geist:wght@400;500;600&family=Geist+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    :root { --color-paper: #f6f4ef; --color-ink: #1b2f4b; --color-muted: #3d5a80; --color-soft: #566880; --color-rule: rgba(27,47,75,0.12); --color-accent: #1f6f66;
      --font-sans: 'Geist', system-ui, sans-serif; --font-serif: 'Instrument Serif', serif; --font-mono: 'Geist Mono', ui-monospace, monospace; }
    body { font-family: var(--font-sans); background: var(--color-paper); min-height: 100vh; padding: 3rem 2rem; color: var(--color-ink); }
    .container { max-width: 1280px; margin: 0 auto; }
    .header { margin-bottom: 1.5rem; }
    .header-eyebrow { font-family: var(--font-mono); font-size: 0.66rem; font-weight: 500; letter-spacing: 0.18em; text-transform: uppercase; color: var(--color-muted); margin-bottom: 0.75rem; }
    h1 { font-family: var(--font-serif); font-size: clamp(1.75rem, 3vw + 1rem, 2.6rem); font-weight: 400; letter-spacing: -0.02em; line-height: 1.1; margin-bottom: 0.6rem; }
    .subtitle { font-size: 1rem; line-height: 1.55; color: var(--color-muted); max-width: 70ch; }
    .diagram-container { width: 100%; overflow-x: auto; }
    svg { width: 100%; min-width: 1280px; display: block; }
    code { font-family: var(--font-mono); font-size: 0.8rem; color: var(--color-ink); }
    .footer { margin-top: 2rem; padding-top: 1.5rem; border-top: 1px solid rgba(27,47,75,0.10); font-family: var(--font-mono); font-size: 0.72rem; letter-spacing: 0.06em; color: var(--color-soft); display: flex; justify-content: space-between; flex-wrap: wrap; gap: 0.5rem; }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <p class="header-eyebrow">Scoring pipeline · PHYS 3153 Project 10</p>
      <h1>How one electrode gets its score</h1>
      <p class="subtitle">The path every electrode takes through the code, from two numbers in a paper to a solar-cell efficiency. Stage 05 is where the two devices differ: a silicon cell collects current with metal fingers about 2&nbsp;mm apart, a perovskite module makes it cross the whole stripe.</p>
    </div>
    <div class="diagram-container">
SVG
    </div>
    <div class="footer">
      <span>transparent-electrode simulation · scoring pipeline</span>
      <span>Brandon Deonarine · UWI St. Augustine · 2026</span>
    </div>
  </div>
</body>
</html>
"""

if __name__ == "__main__":
    open(sys.argv[1], "w", encoding="utf-8").write(HTML.replace("SVG", svg("scoring-pipeline")))
    print("wrote", sys.argv[1])
