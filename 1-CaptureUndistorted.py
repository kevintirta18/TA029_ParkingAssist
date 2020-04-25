import cv2
import numpy as np 
import datetime
import time
import yaml

# Take the undistorted shot from cameras for bird-eye projection-matrix calculation

input_camera_num = 1 
W, H = 640, 480
capture = cv2.VideoCapture(0)

# Load the homography matrix for distortion correction
with open("./yaml/camera1.yaml", "r") as f:
    data = yaml.load(f, Loader=yaml.SafeLoader)

K = np.array(data["K"])
D = np.array(data["D"])

new_K = K.copy()
new_K[0, 0] *= 1.0
new_K[1, 1] *= 1.0

# Make the maps with K and D loaded from yaml file
map1, map2 = cv2.fisheye.initUndistortRectifyMap(
    K,
    D,
    np.eye(3),
    new_K,
    (W, H),
    cv2.CV_16SC2
)

while (1):
    _, frame = capture.read()
    
    # Undistort the frame using maps provided before
    frame = cv2.remap(
        frame,
        map1,
        map2,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT
    )

    try:
        cv2.imshow("Frame", frame)
    except:
        pass 

    key = cv2.waitKey(1) & 0xFF
    if (key == ord('c')):
        # time_stamp = time.time()
        str_time_stamp = datetime.datetime.fromtimestamp(time_stamp).strftime('%d-%m-%Y_%H-%M-%S')
        # cv2.imwrite("camera"+str(input_camera_num)+"-"+str(str_time_stamp)+".jpg", frame)
        cv2.imwrite("camera"+str(input_camera_num)+".jpg", frame)
    if (key == ord('q')):
        break