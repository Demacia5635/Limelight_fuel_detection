import cv2
import numpy as np

# ──────────────────────────────────────────────
#  TUNING CONSTANTS  ← adjust these for your field
# ──────────────────────────────────────────────
YELLOW_LOW  = (18, 100, 100)   # HSV lower bound for yellow
YELLOW_HIGH = (35, 255, 255)   # HSV upper bound for yellow

MIN_RADIUS  = 8      # smallest ball radius to detect (px)
MAX_RADIUS  = 300    # largest ball radius (px)
MIN_DIST    = 80     # min distance between two ball centers (px)
SMOOTH      = 0.3    # 0.0 = frozen, 1.0 = no smoothing
# ──────────────────────────────────────────────

# Smoothed values that persist between frames
_prev_cx, _prev_cy, _prev_cr = 0, 0, 0

def runPipeline(image, llrobot):
    global _prev_cx, _prev_cy, _prev_cr

    llpython       = [0, 0, 0, 0, 0, 0, 0, 0]
    largestContour = np.array([[]])

    # ── 1. Colour mask ────────────────────────────────────────────────
    hsv  = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, YELLOW_LOW, YELLOW_HIGH)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode (mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # ── 2. HoughCircles on the colour mask ───────────────────────────
    blurred = cv2.GaussianBlur(mask, (9, 9), 2)
    hough   = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp        = 1,
        minDist   = MIN_DIST,
        param1    = 50,
        param2    = 25,
        minRadius = MIN_RADIUS,
        maxRadius = MAX_RADIUS
    )

    # ── 3. Pick largest circle = closest ball ─────────────────────────
    cx, cy, cr = 0, 0, 0

    if hough is not None:
        circles = np.round(hough[0]).astype(int)
        best    = max(circles, key=lambda c: c[2])
        cx, cy, cr = int(best[0]), int(best[1]), int(best[2])

    # ── 4. Smooth position using EMA ──────────────────────────────────
    if cr > 0:
        cx = int(_prev_cx + SMOOTH * (cx - _prev_cx))
        cy = int(_prev_cy + SMOOTH * (cy - _prev_cy))
        cr = int(_prev_cr + SMOOTH * (cr - _prev_cr))
        _prev_cx, _prev_cy, _prev_cr = cx, cy, cr
    else:
        _prev_cx = int(_prev_cx + SMOOTH * (0 - _prev_cx))
        _prev_cy = int(_prev_cy + SMOOTH * (0 - _prev_cy))
        _prev_cr = int(_prev_cr + SMOOTH * (0 - _prev_cr))
        cx, cy, cr = _prev_cx, _prev_cy, _prev_cr

    # ── 5. Draw and build contour for Limelight crosshair ─────────────
    if cr > 0:
        cv2.circle(image, (cx, cy), cr, (0, 0, 255), 2)
        cv2.drawMarker(image, (cx, cy), (0, 0, 255),
                       cv2.MARKER_CROSS, 20, 2)
        cv2.putText(image, f"Target: ({cx},{cy}) r={cr}",
                    (5, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 255, 255), 2)

        # Build contour so Limelight crosshair locks onto the ball
        angles = np.linspace(0, 2 * np.pi, 36)
        largestContour = np.array(
            [[[int(cx + cr * np.cos(a)), int(cy + cr * np.sin(a))]]
             for a in angles],
            dtype=np.int32
        )

        # [found, cx, cy, radius, 0, 0, 0, 0]
        llpython = [1, cx, cy, cr, 0, 0, 0, 0]

    return largestContour, image, llpython
