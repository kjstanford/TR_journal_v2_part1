"""Convert simple, single-run <text> elements in a MATLAB-exported SVG into
real vector path outlines.

MATLAB's exportgraphics keeps hand-built text() runs (as opposed to its own
xlabel/ylabel/title objects) as live <text> elements even when FontName
doesn't resolve to an actually-installed font -- which then renders BLANK in
SVG viewers that can't substitute the declared font-family (verified with
resvg). Baking the glyphs into paths here makes the output immune to that,
matching the rest of a MATLAB SVG export (whose more complex/formatted text
already comes out as outlined paths).

Only matches the specific single-string <text> pattern produced by
place_kerned_label() in fig1_gen.m: a <g> with font-family/font-size/
font-weight, wrapping one <text x="0" y="0" ...>STRING</text>. General
formatted MATLAB labels (multi-tspan, etc.) are left untouched.
"""
import re
import subprocess
import sys

from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath


def resolve_font(weight, font_family):
    query = f"{font_family}:bold" if weight == "700" else font_family
    out = subprocess.check_output(["fc-match", "--format=%{file}", query])
    return out.decode().strip()


def path_to_d(path, dx, dy):
    # path.vertices are in font units (y-up, origin at baseline). Flip y
    # (SVG is y-down) and translate to the text anchor (dx, dy).
    d = []
    for verts, code in path.iter_segments():
        xs = verts[0::2]
        ys = verts[1::2]
        x = [dx + v for v in xs]
        y = [dy - v for v in ys]
        if code == 1:  # MOVETO
            d.append(f"M{x[0]:.3f},{y[0]:.3f}")
        elif code == 2:  # LINETO
            d.append(f"L{x[0]:.3f},{y[0]:.3f}")
        elif code == 3:  # CURVE3 (quadratic)
            d.append(f"Q{x[0]:.3f},{y[0]:.3f} {x[1]:.3f},{y[1]:.3f}")
        elif code == 4:  # CURVE4 (cubic)
            d.append(f"C{x[0]:.3f},{y[0]:.3f} {x[1]:.3f},{y[1]:.3f} {x[2]:.3f},{y[2]:.3f}")
        elif code == 79:  # CLOSEPOLY
            d.append("Z")
    return "".join(d)


PATTERN = re.compile(
    r'(<g [^>]*font-family="(?P<family>[^"]+)"[^>]*font-size="(?P<size>[0-9.]+)"'
    r'[^>]*font-weight="(?P<weight>[0-9]+)"[^>]*>\s*)'
    r'<text fill="(?P<fill>[^"]*)"[^>]*x="(?P<x>[0-9.-]+)" y="(?P<y>[0-9.-]+)"[^>]*>'
    r'\s*(?P<str>[^<]*)</text>'
)


def convert(svg_path):
    with open(svg_path) as f:
        content = f.read()

    font_cache = {}

    def repl(m):
        key = (m.group("weight"), m.group("family"))
        if key not in font_cache:
            font_cache[key] = resolve_font(*key)
        fp = FontProperties(fname=font_cache[key])
        tp = TextPath((0, 0), m.group("str"), size=float(m.group("size")), prop=fp)
        d = path_to_d(tp, float(m.group("x")), float(m.group("y")))
        path_el = f'<path fill="{m.group("fill")}" stroke="none" d="{d}"/>'
        return m.group(1) + path_el

    new_content, n = PATTERN.subn(repl, content)
    print(f"text_to_paths: converted {n} text run(s) to outlines in {svg_path}", file=sys.stderr)
    with open(svg_path, "w") as f:
        f.write(new_content)


if __name__ == "__main__":
    convert(sys.argv[1])
