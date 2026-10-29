"""
Make the printable ArUco marker sheet (4 tiles, one per mat corner).

    python make_markers.py            ->  marker_sheet.png   (A4/Letter-safe, 300 dpi)

Print at 100% / "actual size" (NOT "fit to page"). Measure the black square: it must be 20 mm.
Cut out the 4 tiles and glue/tape one at each corner of the mat, flat and wrinkle-free:
    id 0 = top-left, id 1 = top-right, id 2 = bottom-right, id 3 = bottom-left
(top = the side where the phone stand / reference area is, same side you always use).
Keep the white border on each tile: its white paper is also a handy white-balance patch.
"""
import cv2
import numpy as np

DPI = 300
MM = DPI / 25.4
MARKER_MM = 20
BORDER_MM = 15           # white quiet zone around each marker
dic = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

tile = int(round((MARKER_MM + 2 * BORDER_MM) * MM))
mk = int(round(MARKER_MM * MM))
sheet = np.full((int(210 * MM), int(210 * MM), 3), 255, np.uint8)   # fits A4 and Letter
for i in range(4):
    m = cv2.aruco.generateImageMarker(dic, i, mk)
    t = np.full((tile, tile), 255, np.uint8)
    o = (tile - mk) // 2
    t[o:o + mk, o:o + mk] = m
    t = cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(t, (0, 0), (tile - 1, tile - 1), (160, 160, 160), 1)   # cut line
    cv2.putText(t, f"id {i}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 120), 1)
    r, c = divmod(i, 2)
    y, x = int(10 * MM) + r * (tile + int(8 * MM)), int(10 * MM) + c * (tile + int(8 * MM))
    sheet[y:y + tile, x:x + tile] = t
cv2.imwrite("marker_sheet.png", sheet)
try:
    from PIL import Image
    Image.open("marker_sheet.png").save("marker_sheet.png", dpi=(DPI, DPI))
except ImportError:
    pass
print("Wrote marker_sheet.png - print at 100% scale; black marker square must measure 20 mm.")
