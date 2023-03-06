import cv2
import time
import numpy as np

# parameters
width = 640
height = 480
brightness = 50

# read webcam
webcam = cv2.VideoCapture(0)

#set parameters
webcam.set(3,width)
webcam.set(4,height)
webcam.set(10,brightness)

points = []

colors = [ # BLUE
    [81, 125, 255, 127, 255, 255],
    # YELLOW
    [24,86, 194, 65, 234, 255]
]

drawColor  = [
    (255,0,0),
    (0,255,255)
]

# find shape and get points
def find_shape_get_points(maskedFrame):
    global points
    # maskedFrame -> gray -> blur -> edges
    frameGray = cv2.cvtColor(maskedFrame,cv2.COLOR_BGR2GRAY)
    frameBlur = cv2.GaussianBlur(frameGray,(7,7),1)
    frameEdges = cv2.Canny(frameBlur,50,50)

    # find contours
    contours, hierarchy = cv2.findContours(frameEdges,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # if area of contour is more than 200, then that contour is important for us
        if area > 200:
            perimeter = cv2.arcLength(cnt,True)
            corners = cv2.approxPolyDP(cnt,0.2 * perimeter, True)
            # P.S:
            # i would like to use number_corners and if it is equal 10 (that means its circle) then we will draw it
            # but the problem of quality of our webcam and not good edge defining cant let me to do that
            # so we can not use number_corners
            number_corners = len(corners)
            # find bounding rectangle
            x_rec, y_rec, width_rec, height_rec = cv2.boundingRect(cnt)

            center_x, center_y = int(x_rec+(width_rec/2)),int(y_rec+(height_rec/2)) 
            points.append([center_x,center_y,colors.index(color)])


# reading and showing frames
while True:
    # read each frame
    success,frame = webcam.read()
    
    # SECTION : color detection
    frameHsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    for color in colors:
        # define low bound
        lower = np.array(
            color[0:3] # h_min, s_min, v_min
        )

        # define up bound
        upper = np.array(
            color[3:6] # h_max, s_max, v_max
        )

        # build mask
        mask = cv2.inRange(frameHsv,lower,upper)

        # result
        maskedFrame = cv2.bitwise_and(frame,frame,mask = mask)

        # SECTION : find shape
        find_shape_get_points(maskedFrame)

    # draw points
    if len(points) != 0:
        for point in points:
            cv2.circle(frame,[point[0], point[1]],10,drawColor[point[2]],cv2.FILLED)

    # show
    cv2.imshow("Result", frame)
    if cv2.waitKey(1) == ord('q'):
        # quit
        break
    elif cv2.waitKey(1) == ord('c'):
        # clean marker
        points = []