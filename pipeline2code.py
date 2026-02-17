import cv2
import numpy as np
from scipy import ndimage

def runPipeline(image, llrobot):
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    img_threshold = cv2.inRange(img_hsv, (20, 120, 50), (77, 255, 255))

    kernel = np.ones((5, 5), np.uint8)
    img_threshold = cv2.erode(img_threshold, kernel, iterations=1)
    img_threshold = cv2.dilate(img_threshold, kernel, iterations=2)

    # --- Tunable thresholds ---
    MIN_AREA = 200
    MAX_AREA = 8000
    MIN_CIRCULARITY = 0.45   # slightly relaxed since watershed splits are less perfect
    DIST_THRESHOLD = 0.45    # higher = stricter separation (0.3–0.6 range)

    # -------------------------------------------------------
    # STEP 1: Distance transform to find ball centers
    # Each pixel's value = distance to nearest background pixel
    # Ball centers will be local peaks in this map
    # -------------------------------------------------------
    dist_transform = cv2.distanceTransform(img_threshold, cv2.DIST_L2, 5)
    cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)

    # Threshold the distance map — only keep the "sure foreground" peaks
    _, sure_fg = cv2.threshold(dist_transform, DIST_THRESHOLD, 1.0, cv2.THRESH_BINARY)
    sure_fg = np.uint8(sure_fg)

    # -------------------------------------------------------
    # STEP 2: Label each separated peak as a unique marker
    # These become the "seeds" for watershed
    # -------------------------------------------------------
    num_labels, markers = cv2.connectedComponents(sure_fg)

    # Add 1 so background is 1, not 0 (watershed needs 0 for unknown regions)
    markers = markers + 1

    # Mark the unknown region (between sure_fg and the full threshold) as 0
    sure_bg = cv2.dilate(img_threshold, kernel, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)
    markers[unknown == 255] = 0

    # -------------------------------------------------------
    # STEP 3: Run watershed — it floods from each seed marker
    # and draws boundaries where regions meet
    # -------------------------------------------------------
    image_3ch = image.copy()
    cv2.watershed(image_3ch, markers)

    # -------------------------------------------------------
    # STEP 4: Extract each watershed region as a candidate ball
    # -------------------------------------------------------
    ball_rects = []

    for label in range(2, num_labels + 1):  # skip 1 (background) and -1 (boundaries)
        # Create a mask for this single region
        mask = np.uint8(markers == label) * 255

        region_contours, _ = cv2.findContours(mask,
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not region_contours:
            continue

        c = max(region_contours, key=cv2.contourArea)
        area = cv2.contourArea(c)

        if area < MIN_AREA or area > MAX_AREA:
            continue

        # Circularity check per separated region
        perimeter = cv2.arcLength(c, True)
        if perimeter == 0:
            continue
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        if circularity < MIN_CIRCULARITY:
            continue

        x, y, w, h = cv2.boundingRect(c)
        ball_rects.append((x, y, w, h))

        # Draw each detected ball in green
        cv2.drawContours(image, [c], -1, (0, 255, 0), 2)

    # -------------------------------------------------------
    # STEP 5: Pick the closest ball (lowest bottom edge)
    # -------------------------------------------------------
    llpython = [0, 0, 0, 0, 0, 0, 0, 0]
    largestContour = np.array([[]])

    if len(ball_rects) > 0:
        best = max(ball_rects, key=lambda r: r[1] + r[3])
        x, y, w, h = best

        # Highlight target
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 3)
        cx, cy = x + w // 2, y + h // 2
        cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)
        cv2.putText(image, f"TARGET ({len(ball_rects)} balls found)",
                    (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        largestContour = np.array([[[x, y]], [[x+w, y]],
                                    [[x+w, y+h]], [[x, y+h]]])
        llpython = [1, x, y, w, h, cx, cy, len(ball_rects)]

    return largestContour, image, llpython
