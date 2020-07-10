import numpy as np
import cv2
import yaml
import os
from paramsettings import *

# Camera1 = Front-Right
# Camera2 = Rear-Right
# Camera3 = Rear-Left
# Camera4 = Front-Left

parkir_cascade = cv2.CascadeClassifier("./cascade.xml")

# Getting camera's parameters and example images
W, H = 640, 480
work_dir = "./yaml"
car_image = os.path.join(work_dir, "car.jpg")
camera_params = [os.path.join(work_dir, f) for f in ("camera1.yaml", "camera2.yaml", "camera3.yaml", "camera4.yaml")]
camera_images = [os.path.join(work_dir, f) for f in ("camera1.jpg", "camera2.jpg", "camera3.jpg", "camera4.jpg")]

counter = 16
x0, y0, x1, y1, x2, y2 = 0, 0, 0, 0, 0, 0
is_detected = 0
refPt = []
lower = [0, 0, 0]
upper = [0, 0, 0]

def main(save_weights=False):
    PATH = './'

    matrices = []
    undistort_maps = []
    for conf in camera_params:
        with open(conf, "r") as f:
            data = yaml.load(f, Loader=yaml.SafeLoader)

        proj_mat = np.array(data["M"])
        matrices.append(proj_mat)

        K = np.array(data["K"])
        D = np.array(data["D"])
        scale = np.array(data["scale"])
        new_K = K.copy()
        new_K[0, 0] *= scale[0]
        new_K[1, 1] *= scale[1]
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K,
            D,
            np.eye(3),
            new_K,
            (W, H),
            cv2.CV_16SC2
        )
        undistort_maps.append((map1, map2))

    images = [cv2.imread(im) for im in camera_images]

    # From Video
    # capture_1 = cv2.VideoCapture(PATH+'out_1.mp4')
    # capture_2 = cv2.VideoCapture(PATH+'out_2.mp4')
    # capture_3 = cv2.VideoCapture(PATH+'out_3.mp4')
    # capture_4 = cv2.VideoCapture(PATH+'out_4.mp4')

    # From camera
    # Cameras' indexes must be assigned manually
    capture_1 = cv2.VideoCapture(0)
    capture_2 = cv2.VideoCapture(2)
    capture_3 = cv2.VideoCapture(4)
    capture_4 = cv2.VideoCapture(6)
    
    car = cv2.imread(car_image)
    car = cv2.resize(car, (143, 198))
    
    is_configured = 0

while (True):
        _, frame_1 = capture_1.read()
        _, frame_2 = capture_2.read()
        _, frame_3 = capture_3.read()
        _, frame_4 = capture_4.read()


        map1, map2 = undistort_maps[0]
        frame_1 = cv2.remap(frame_1, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        map1, map2 = undistort_maps[1]
        frame_2 = cv2.remap(frame_2, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        map1, map2 = undistort_maps[2]
        frame_3 = cv2.remap(frame_3, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        map1, map2 = undistort_maps[3]
        frame_4 = cv2.remap(frame_4, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        camera1 = cv2.warpPerspective(
            frame_1,
            matrices[0],
            sideShape
        )
        camera2 = cv2.warpPerspective(
            frame_2,
            matrices[1],
            rearShape
        )
        camera3 = cv2.warpPerspective(
            frame_3,
            matrices[2],
            rearShape
        )
        camera4 = cv2.warpPerspective(
            frame_4,
            matrices[3],
            sideShape
        )

        camera1 = np.flip(np.flip(cv2.transpose(camera1)[::-1], 1),0)
        camera2 = camera2[::-1, ::-1, :]
        camera3 = camera3[::-1, ::-1, :]
        camera4 = np.flip(cv2.transpose(camera4), 0)

        camera2b = camera2[50:170,     350:418]
        camera3b = camera3[50:170,     395:470]
        camera4b = camera4[110:380, sideShape[1]-180:]

        camera2b = cv2.resize(camera2b, (camera2b.shape[1], camera2b.shape[0]*3//5))
        camera3b = cv2.resize(camera3b, (camera3b.shape[1], camera3b.shape[0]*3//5))
        camera1b = cv2.resize(camera1, (camera1.shape[1], int(camera1.shape[0]*1.1)))
        camera1b = camera1b[115:385, :180]

        height = camera4b.shape[0]
        width = camera1b.shape[1]+camera2b.shape[1]+camera3b.shape[1]+camera4b.shape[1]
        result = np.zeros((height, width, 3), dtype=np.uint8)

        result[:, :camera4b.shape[1]] = camera4b
        start_x = camera4b.shape[1]

        result[height-camera3b.shape[0]:, start_x:start_x+camera3b.shape[1]] = camera3b
        start_x = start_x+camera3b.shape[1]

        result[height-camera2b.shape[0]:, start_x:start_x+camera2b.shape[1]] = camera2b
        start_x = start_x+camera2b.shape[1]

        result[:, start_x:] = camera1b

        result = cv2.convertScaleAbs(result, alpha = 2.0, beta = 20)           
        result[:car.shape[0], camera4b.shape[1]:camera4b.shape[1]+car.shape[1]] = car
        
        if (not is_configured)
            out_noHaar = cv2.VideoWriter(PATH+'out_noHaar.mp4',cv2.VideoWriter_fourcc('M','J','P','G'), 24, (result.[1], result[0]))
            out_overall = cv2.VideoWriter(PATH+'out_overall.mp4',cv2.VideoWriter_fourcc('M','J','P','G'), 24, (result.[1], result[0]))
            is_configured = 1

        # out_noHaar.write(result) #Save Videos with no detection

        # Haar Detection
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        parkir = parkir_cascade.detectMultiScale(gray, scaleFactor = 1.3, 
                                        minNeighbors = 27, 
                                        minSize=(40,40),
                                        maxSize=(40,40))
        for x, y, w, h in parkir:
            if (((x<180) or (x>324)) or (y>200)):
                result = cv2.rectangle(result, (x,y), (x+w, y+h), (0, 255, 0), 3)

        # out_overall.write(result) #Save Video with Haar detection
        
        cv2.imshow("result", result)
        if (cv2.waitKey(1) & 0xFF == ord('q')):
            break

if __name__ == "__main__":
    main(save_weights=True)
