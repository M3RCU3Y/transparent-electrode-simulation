"""Writes data/fallback.tex: placeholder macros used only when numbers.tex
does not yet contain the Monte Carlo results (\\providecommand never
overrides a real definition)."""
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "fallback.tex")
names = ["ITO", "IOH", "AZO", "FTO", "DMD", "GRA", "AgNW", "PEDOT", "CNT"]
lines = ["% fallbacks used only if numbers.tex lacks Monte Carlo results"]
for a in ("SHJ", "PSK"):
    for n in names:
        for p in ("mc", "mclo", "mchi", "pfirst"):
            lines.append(r"\providecommand{\%s%s%s}{??}" % (p, a, n))
lines.append(r"\providecommand{\Nmc}{??}")
open(OUT, "w").write("\n".join(lines) + "\n")
