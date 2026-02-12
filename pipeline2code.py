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

def drawDecorations(image, tx, ty, circles_count, total_area):
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
    cv2.putText(image, 
        f'Total Area: {total_area:.0f} px', 
        (10, 90), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.7, (255, 128, 0), 2, cv2.LINE_AA)

def find_densest_contour_center(contours, img_shape):
    """Find the center of the area with the most contours and return contributing contours"""
    if len(contours) == 0:
        return 0.0, 0.0, 0, 0, []
    
    # Create a density map
    density_map = np.zeros(img_shape[:2], dtype=np.uint8)
    
    # Draw all contours on density map
    for contour in contours:
        cv2.drawContours(density_map, [contour], -1, 255, -1)
    
    # Find moments to get center of mass
    M = cv2.moments(density_map)
    
    if M["m00"] == 0:
        return 0.0, 0.0, 0, 0, []
    
    # Calculate center coordinates
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    
    # Convert to normalized tx, ty (-1 to 1, with 0,0 at image center)
    height, width = img_shape[:2]
    tx = (cx - width/2) / (width/2)
    ty = (height/2 - cy) / (height/2)
    
    # Find which contours are near the center (contribute to density)
    # A contour contributes if it's within a reasonable distance from center
    contributing_contours = []
    max_distance = min(width, height) * 0.3  # 30% of image size
    
    for contour in contours:
        # Get contour center
        contour_M = cv2.moments(contour)
        if contour_M["m00"] != 0:
            contour_cx = contour_M["m10"] / contour_M["m00"]
            contour_cy = contour_M["m01"] / contour_M["m00"]
            
            # Calculate distance from density center
            distance = np.sqrt((contour_cx - cx)**2 + (contour_cy - cy)**2)
            
            # If close enough, it contributes to the density
            if distance < max_distance:
                contributing_contours.append(contour)
    
    return tx, ty, int(cx), int(cy), contributing_contours

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
    
    # ===== FILTER CONTOURS =====
    # Remove very large contours (robots) and very small contours (noise)
    min_contour_area = 100      # Minimum area in pixels (adjust based on your objects)
    max_contour_area = 10000    # Maximum area in pixels (adjust to exclude robots)
    
    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_contour_area < area < max_contour_area:
            filtered_contours.append(contour)
    
    # Use filtered contours for everything
    contours = filtered_contours
    # ===========================
  
    # Find densest contour area center and get contributing contours
    tx, ty, cx, cy, contributing_contours = find_densest_contour_center(contours, img.shape)
    
    # Calculate total area of ONLY contours that contribute to density
    total_area = 0
    for contour in contributing_contours:
        total_area += cv2.contourArea(contour)
    
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
    # Draw circles and count only those inside CONTRIBUTING contours
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # Check if circle center is inside any CONTRIBUTING contour
            circle_center = (i[0], i[1])
            is_inside_contour = False
            
            for contour in contributing_contours:  # Only check contributing contours
                # Check if point is inside this contour
                result = cv2.pointPolygonTest(contour, circle_center, False)
                if result >= 0:  # Point is inside or on the contour
                    is_inside_contour = True
                    break
            
            # Only count and draw circles that are inside contributing contours
            if is_inside_contour:
                circles_count += 1
                cv2.circle(img, circle_center, i[2], (0, 255, 0), 2)  # Green outline
                cv2.circle(img, circle_center, 2, (0, 0, 255), 3)      # Red center
            else:
                # Draw rejected circles in gray for debugging
                cv2.circle(img, circle_center, i[2], (128, 128, 128), 1)
                cv2.circle(img, circle_center, 2, (128, 128, 128), 2)
    
    # Draw contours - different colors for contributing vs non-contributing
    # Non-contributing contours in light blue (detected but not part of cluster)
    for contour in contours:
        if contour not in contributing_contours:
            cv2.drawContours(img, [contour], -1, (255, 200, 100), 1)
    
    # Contributing contours in bright blue (actively part of the density cluster)
    cv2.drawContours(img, contributing_contours, -1, (255, 0, 0), 2)
    
    # Draw center of densest area (yellow crosshair)
    if tx != 0 or ty != 0:
        cv2.circle(img, (cx, cy), 10, (0, 255, 255), -1)
        cv2.circle(img, (cx, cy), 15, (0, 255, 255), 2)
        cv2.line(img, (cx-20, cy), (cx+20, cy), (0, 255, 255), 2)
        cv2.line(img, (cx, cy-20), (cx, cy+20), (0, 255, 255), 2)
  
    incrementTestVar()
    drawDecorations(img, tx, ty, circles_count, total_area)
    
    # Store tx, ty, circle count, and total area in llpython array
    llpython = [tx, ty, circles_count, total_area, 0, 0, 0, 0]
       
    return largestContour, img, llpython
