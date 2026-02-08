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
    # Get parameters from Limelight's built-in sliders
    # Access via llrobot dictionary - these map to the sliders in the web UI
    minDist = llrobot.get(0, 30)      # Slider 0
    param1 = llrobot.get(1, 50)        # Slider 1
    param2 = llrobot.get(2, 20)        # Slider 2
    minRadius = llrobot.get(3, 5)      # Slider 3
    maxRadius = llrobot.get(4, 150)    # Slider 4
    
    hue_min = llrobot.get(5, 20)       # Slider 5
    hue_max = llrobot.get(6, 77)       # Slider 6
    sat_min = llrobot.get(7, 120)      # Slider 7
    sat_max = llrobot.get(8, 255)      # Slider 8
    val_min = llrobot.get(9, 50)       # Slider 9
    val_max = llrobot.get(10, 255)     # Slider 10
    
    # Ensure valid values
    minDist = max(1, int(minDist))
    param2 = max(1, int(param2))
    
    img = cv2.medianBlur(image, 5)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Use parameters for color threshold
    img_threshold = cv2.inRange(img_hsv, (int(hue_min), int(sat_min), int(val_min)), 
                                (int(hue_max), int(sat_max), int(val_max)))
   
    contours, _ = cv2.findContours(img_threshold, 
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  
    largestContour = np.array([[]])
    
    # Find densest contour area center
    tx, ty, cx, cy = find_densest_contour_center(contours, img.shape)
    
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
