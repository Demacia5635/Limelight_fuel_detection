import cv2
import numpy as np

# global variables go here:
testVar = 0

# To change a global variable inside a function,
# re-declare it with the 'global' keyword
def incrementTestVar():
    global testVar
    testVar = testVar + 1
    if testVar == 100:
        print("test")
    if testVar >= 200:
        print("print")
        testVar = 0

def drawDecorations(image):
    cv2.putText(image, 
        'Limelight python script!', 
        (0, 230), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        .5, (0, 255, 0), 1, cv2.LINE_AA)
    
# runPipeline() is called every frame by Limelight's backend.
def runPipeline(image, llrobot):
    img = cv2.medianBlur(image, 5)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_threshold = cv2.inRange(img_hsv, (20,120,50), (77,260,255))
   
    contours, _ = cv2.findContours(img_threshold, 
    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  
    largestContour = np.array([[]])
    llpython = [0,0,0,0,0,0,0,0]

    # Use grayscale image for HoughCircles
    circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, 
                            dp=1, 
                            minDist=50,      # Increased: min distance between circle centers
                            param1=100,      # Increased: higher threshold for edge detection
                            param2=30,       # Accumulator threshold (lower = more circles)
                            minRadius=10,    # Set actual minimum radius
                            maxRadius=100)   # Set actual maximum radius
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.circle(img, (i[0], i[1]), 2, (0, 255, 0), 3)
  
    incrementTestVar()
    drawDecorations(img)  # Also changed 'image' to 'img' for consistency
       
    return largestContour, img, llpython
