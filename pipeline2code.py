import cv2
import numpy as np

testVar = 0

def incrementTestVar():
    global testVar
    testVar = testVar + 1
    if testVar == 100:
        print("test")
    if testVar >= 200:
        print("print")
        testVar = 0

def drawDecorations(image, tx, ty, circles_count):
    cv2.putText(image, 
        f'tx: {tx:.2f}, ty: {ty:.2f}', 
        (10, 30), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.7, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, 
        f'Circles: {circles_count}', 
        (10, 60), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.7, (0, 255, 0), 2, cv2.LINE_AA)

def find_densest_contour_center(contours, img_shape):
    """Find the center of the area with the most contours"""
    if len(contours) == 0:
        return 0.0, 0.0, 0, 0
    
    # Create a density map
    density_map = np.zeros(img_shape[:2], dtype=np.uint8)
    
    # Draw all contours on density map
    for contour in contours:
        cv2.drawContours(density_map, [contour], -1, 255, -1)
    
    # Find moments to get center of mass
    M = cv2.moments(density_map)
    
    if M["m00"] == 0:
        return 0.0, 0.0, 0, 0
    
    # Calculate center coordinates
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    
    # Convert to normalized tx, ty (-1 to 1, with 0,0 at image center)
    height, width = img_shape[:2]
    tx = (cx - width/2) / (width/2)
    ty = (height/2 - cy) / (height/2)
    
    return tx, ty, int(cx), int(cy)

def runPipeline(image, llrobot):
    # ===== TUNABLE PARAMETERS - Adjust these values =====
    # Circle detection parameters
    minDist = 30        # Minimum distance between circle centers
    param1 = 50         # Canny edge detection threshold
    param2 = 20         # Circle detection threshold (lower = more circles)
    minRadius = 5       # Minimum circle radius in pixels
    maxRadius = 150     # Maximum circle radius in pixels
    
    # HSV color filtering parameters
    # Hue: 0-179 (color), Saturation: 0-255 (color intensity), Value: 0-255 (brightness)
    hue_min = 20        # Minimum hue (e.g., 20-40 = yellow/orange)
    hue_max = 77        # Maximum hue
    sat_min = 120       # Minimum saturation
    sat_max = 255       # Maximum saturation
    val_min = 50        # Minimum brightness
    val_max = 255       # Maximum brightness
    # ====================================================
    
    img = cv2.medianBlur(image, 5)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Use parameters for color threshold
    img_threshold = cv2.inRange(img_hsv, (int(hue_min), int(sat_min), int(val_min)), 
                                (int(hue_max), int(sat_max), int(val_max)))
   
    contours, _ = cv2.findContours(img_threshold, 
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  
    # Find densest contour area center
    tx, ty, cx, cy = find_densest_contour_center(contours, img.shape)
    
    # Create a synthetic contour at the calculated center point
    # This will be a small rectangle centered at (cx, cy)
    # Limelight will use this contour's center for tx/ty calculations
    contour_size = 10  # Size of the synthetic contour
    largestContour = np.array([
        [[cx - contour_size, cy - contour_size]],
        [[cx + contour_size, cy - contour_size]],
        [[cx + contour_size, cy + contour_size]],
        [[cx - contour_size, cy + contour_size]]
    ], dtype=np.int32)
    
    # Detect circles
    circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, 
                              dp=1, 
                              minDist=int(minDist),
                              param1=int(param1),
                              param2=int(param2),
                              minRadius=int(minRadius),
                              maxRadius=int(maxRadius))

    circles_count = 0
    # Draw circles
    if circles is not None:
        circles_count = len(circles[0])
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)
    
    # Draw contours
    cv2.drawContours(img, contours, -1, (255, 0, 0), 2)
    
    # Draw center of densest area (yellow crosshair)
    if tx != 0 or ty != 0:
        cv2.circle(img, (cx, cy), 10, (0, 255, 255), -1)
        cv2.circle(img, (cx, cy), 15, (0, 255, 255), 2)
        cv2.line(img, (cx-20, cy), (cx+20, cy), (0, 255, 255), 2)
        cv2.line(img, (cx, cy-20), (cx, cy+20), (0, 255, 255), 2)
  
    incrementTestVar()
    drawDecorations(img, tx, ty, circles_count)
    
    # Store tx, ty, and circle count in llpython array
    llpython = [tx, ty, circles_count, 0, 0, 0, 0, 0]
       
    return largestContour, img, llpython
