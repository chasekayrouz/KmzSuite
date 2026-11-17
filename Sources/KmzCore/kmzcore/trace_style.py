"""Session visualization colors. No Qt, filesystem access or persistence."""
from colorsys import hsv_to_rgb
from functools import lru_cache
from math import pow

LINE_WIDTH = 1.7
FILL_ALPHA = 0.065


def _rgb(color):
    return tuple(int(color[i:i + 2], 16) / 255 for i in (1, 3, 5))


def _linear(value):
    return value / 12.92 if value <= 0.04045 else pow((value + 0.055) / 1.055, 2.4)


def _lab(color):
    r, g, b = map(_linear, _rgb(color))
    x = (r * .4124564 + g * .3575761 + b * .1804375) / .95047
    y = r * .2126729 + g * .7151522 + b * .0721750
    z = (r * .0193339 + g * .1191920 + b * .9503041) / 1.08883
    def f(v):
        return v ** (1 / 3) if v > (6 / 29) ** 3 else v / (3 * (6 / 29) ** 2) + 4 / 29
    return (116 * f(y) - 16, 500 * (f(x) - f(y)), 200 * (f(y) - f(z)))


@lru_cache(maxsize=1)
def chromatogram_palette():
    """24 distinct colors, greedy maximin CIELAB spacing before palette reuse.

    Saturated candidates must have at least 3:1 contrast with the white plot.
    Each next color maximizes its minimum distance to ALL preceding colors,
    avoiding the neighboring pale/dark pairs common in categorical palettes.
    Many simultaneous colors still require source labels; no finite palette
    provides unlimited distinction or universal color-vision accessibility.
    """
    colors = ["#0072b2", "#d55e00", "#009e73", "#cc3377"]
    candidates = set()
    for hue in range(0, 360, 5):
        for saturation in (.60, .80, 1.0):
            for value in (.65, .82, 1.0):
                rgb = hsv_to_rgb(hue / 360, saturation, value)
                color = "#" + "".join(f"{round(c * 255):02x}" for c in rgb)
                luminance = sum(w * _linear(c) for w, c in zip((.2126, .7152, .0722), _rgb(color)))
                if 1.05 / (luminance + .05) >= 3:
                    candidates.add(color)
    points = {color: _lab(color) for color in sorted(candidates.difference(colors))}
    # Running minima hold each candidate's exact distance to every chosen color;
    # min is order-independent, and max keeps the first tie in candidate order.
    nearest = dict.fromkeys(points, float("inf"))
    added = colors
    while len(colors) < 24:
        for other in map(_lab, added):
            for color, point in points.items():
                nearest[color] = min(nearest[color], sum((a - b) ** 2 for a, b in zip(point, other)))
        chosen = max(points, key=nearest.__getitem__)
        colors.append(chosen)
        added = [chosen]
        del points[chosen]
    return tuple(colors)


def fill_rgba(color):
    """The base hue with a faint alpha, independent of line emphasis."""
    return (*tuple(round(c * 255) for c in _rgb(color)), round(255 * FILL_ALPHA))


def line_style(color, emphasized=False):
    if not emphasized:
        return color, LINE_WIDTH
    # Darken in RGB, retaining hue; never feed this color into fill_rgba.
    return "#" + "".join(f"{round(c * 255 * .72):02x}" for c in _rgb(color)), LINE_WIDTH * 2
