# Draws the package icon: a small network on a dark rounded square, in the two colours the
# documentation site uses. Committed alongside the PNG it produces, so the icon can be changed by
# changing a number here rather than by opening a drawing program nobody has.
#
#   python tools/icon/icon.py            writes assets/icon.png at 128x128
#
# It draws at six times the size and averages down, which is where the smooth edges come from: a
# circle of exactly the right radius at 128 pixels has a staircase along it, and the same circle
# drawn at 768 and averaged has a soft one.

import math
import struct
import zlib
from pathlib import Path

SIZE = 128
SCALE = 6
BIG = SIZE * SCALE

GROUND_TOP = (0x14, 0x24, 0x3F)
GROUND_BOTTOM = (0x08, 0x11, 0x1F)
EDGE = (0x38, 0xBD, 0xF8)
NODE = (0x67, 0xE8, 0xF9)
ANSWER = (0xF1, 0xF5, 0xF9)

# Three layers, left to right: what goes in, what it is turned into, what comes out.
LAYERS = [3, 4, 2]
COLUMNS = [0.235, 0.5, 0.765]


def blend(under, over, alpha):
    return tuple(round(u + (o - u) * alpha) for u, o in zip(under, over))


def rounded_square(pixels):
    radius = BIG * 0.225

    for y in range(BIG):
        shade = blend(GROUND_TOP, GROUND_BOTTOM, y / (BIG - 1))

        for x in range(BIG):
            near_x = min(max(x + 0.5, radius), BIG - radius)
            near_y = min(max(y + 0.5, radius), BIG - radius)

            if math.hypot(x + 0.5 - near_x, y + 0.5 - near_y) > radius:
                continue

            at = (y * BIG + x) * 4
            pixels[at:at + 4] = bytes(shade) + b"\xff"


def positions():
    places = []

    for layer, count in enumerate(LAYERS):
        column = COLUMNS[layer] * BIG
        span = 0.62 * BIG
        step = span / (count - 1) if count > 1 else 0

        top = (BIG - step * (count - 1)) / 2
        places.append([(column, top + step * row) for row in range(count)])

    return places


def line(pixels, start, end, width, colour, alpha):
    (x1, y1), (x2, y2) = start, end
    length = math.hypot(x2 - x1, y2 - y1)
    half = width / 2

    for y in range(int(min(y1, y2) - half), int(max(y1, y2) + half) + 1):
        for x in range(int(min(x1, x2) - half), int(max(x1, x2) + half) + 1):
            if not (0 <= x < BIG and 0 <= y < BIG):
                continue

            along = ((x + 0.5 - x1) * (x2 - x1) + (y + 0.5 - y1) * (y2 - y1)) / (length * length)
            along = min(max(along, 0), 1)

            if math.hypot(x + 0.5 - (x1 + along * (x2 - x1)), y + 0.5 - (y1 + along * (y2 - y1))) > half:
                continue

            at = (y * BIG + x) * 4
            under = tuple(pixels[at:at + 3])
            pixels[at:at + 4] = bytes(blend(under, colour, alpha)) + b"\xff"


def disc(pixels, centre, radius, colour):
    cx, cy = centre

    for y in range(int(cy - radius), int(cy + radius) + 1):
        for x in range(int(cx - radius), int(cx + radius) + 1):
            if not (0 <= x < BIG and 0 <= y < BIG) or math.hypot(x + 0.5 - cx, y + 0.5 - cy) > radius:
                continue

            at = (y * BIG + x) * 4
            pixels[at:at + 4] = bytes(colour) + b"\xff"


def shrink(pixels):
    small = bytearray(SIZE * SIZE * 4)
    squares = SCALE * SCALE

    for y in range(SIZE):
        for x in range(SIZE):
            totals = [0, 0, 0, 0]

            for dy in range(SCALE):
                row = ((y * SCALE + dy) * BIG + x * SCALE) * 4

                for dx in range(SCALE):
                    at = row + dx * 4
                    for channel in range(4):
                        totals[channel] += pixels[at + channel]

            at = (y * SIZE + x) * 4
            small[at:at + 4] = bytes(total // squares for total in totals)

    return small


def png(pixels):
    raw = b"".join(b"\x00" + bytes(pixels[y * SIZE * 4:(y + 1) * SIZE * 4]) for y in range(SIZE))

    def chunk(kind, body):
        return (struct.pack(">I", len(body)) + kind + body
                + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF))

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", SIZE, SIZE, 8, 6, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9))
            + chunk(b"IEND", b""))


def main():
    pixels = bytearray(BIG * BIG * 4)

    rounded_square(pixels)
    places = positions()

    for left, right in zip(places, places[1:]):
        for start in left:
            for end in right:
                line(pixels, start, end, 0.011 * BIG, EDGE, 0.42)

    for layer, column in enumerate(places):
        for centre in column:
            disc(pixels, centre, 0.055 * BIG, ANSWER if layer == len(places) - 1 else NODE)

    out = Path(__file__).resolve().parents[2] / "assets" / "icon.png"
    out.parent.mkdir(exist_ok=True)
    out.write_bytes(png(shrink(pixels)))

    print(f"{out} — {out.stat().st_size} bytes, {SIZE}x{SIZE}")


if __name__ == "__main__":
    main()
