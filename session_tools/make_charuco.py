#!/usr/bin/env python3
"""Generate a print-ready ChArUco board (OpenCV 4.11 aruco API).

Outputs a PNG + a PDF whose embedded DPI makes it print at TRUE physical size
(as long as the printer is set to 100% / "Actual size", NOT "fit to page").

After printing, MEASURE one black square edge-to-edge with calipers and use that
measured value as --square in the calibration step. The print scale is what it
is; the calibrator only needs the real square size to get translation right.

Usage:
  python3 session_tools/make_charuco.py            # defaults: 5x7, 35mm sq, A4
  python3 session_tools/make_charuco.py --squares-x 5 --squares-y 7 \
        --square 35 --marker 26 --dict DICT_4X4_50 --dpi 300 --out calib/charuco
"""
import argparse, os
import numpy as np
import cv2
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--squares-x", type=int, default=5, help="# squares across")
ap.add_argument("--squares-y", type=int, default=7, help="# squares down")
ap.add_argument("--square", type=float, default=35.0, help="square side (mm)")
ap.add_argument("--marker", type=float, default=26.0, help="aruco marker side (mm)")
ap.add_argument("--dict", default="DICT_4X4_50", help="cv2.aruco predefined dict")
ap.add_argument("--dpi", type=int, default=300)
ap.add_argument("--margin", type=float, default=10.0, help="white border (mm)")
ap.add_argument("--out", default="calib/charuco")
a = ap.parse_args()

assert a.marker < a.square, "marker must be smaller than square"
mm2px = lambda mm: int(round(mm / 25.4 * a.dpi))

aruco = cv2.aruco
dic = aruco.getPredefinedDictionary(getattr(aruco, a.dict))
board = aruco.CharucoBoard((a.squares_x, a.squares_y), a.square / 1000.0,
                           a.marker / 1000.0, dic)   # lengths in metres

sq_px, mg_px = mm2px(a.square), mm2px(a.margin)
W = a.squares_x * sq_px + 2 * mg_px
H = a.squares_y * sq_px + 2 * mg_px
img = board.generateImage((W, H), marginSize=mg_px, borderBits=1)

os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
png, pdf = a.out + ".png", a.out + ".pdf"
cv2.imwrite(png, img)
Image.fromarray(img).convert("L").save(pdf, resolution=float(a.dpi))

n_markers = (a.squares_x * a.squares_y) // 2
print(f"ChArUco {a.squares_x}x{a.squares_y}  square={a.square}mm  marker={a.marker}mm")
print(f"dict={a.dict}  markers used={n_markers}  ({a.dict.split('_')[1]} grid, "
      f"dict holds {len(dic.bytesList)})")
print(f"board print size : {a.squares_x*a.square:.0f} x {a.squares_y*a.square:.0f} mm "
      f"(+{a.margin:.0f}mm border)  -> {'A4' if W<mm2px(210) and H<mm2px(297) else 'check paper'}")
print(f"image            : {W}x{H}px @ {a.dpi}dpi")
print(f"saved            : {png}\n                   {pdf}")
print("\nPRINT AT 100% / ACTUAL SIZE. Then measure a square and pass it as --square to the calibrator.")
