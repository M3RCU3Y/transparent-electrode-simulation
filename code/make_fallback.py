# makes data/fallback.tex placeholders so latex compiles before monte carlo finishes
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "fallback.tex")
names = ["ITO", "IOH", "AZO", "FTO", "DMD", "GRA", "AgNW", "PEDOT", "CNT"]
lines = ["% fallback placeholders if numbers.tex does not have monte carlo yet"]
for a in ("SHJ", "PSK"):
    for n in names:
        for p in ("mc", "mclo", "mchi", "pfirst"):
            lines.append(r"\providecommand{\%s%s%s}{??}" % (p, a, n))
lines.append(r"\providecommand{\Nmc}{??}")
open(OUT, "w").write("\n".join(lines) + "\n")
