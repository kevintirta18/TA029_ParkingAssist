"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Stitch four camera images to form the 360 surround view
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
import numpy as np
import cv2
import yaml
import os
from paramsettings import *

# Camera1 = Front-Right
# Camera2 = Rear-Right
# Camera3 = Rear-Left
# Camera4 = Front-Left


class Tracker():
    def __init__(self, x = 0, y = 0, w = 0, h = 0):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

        self.miss_counter = 0

    def track(self, new_x, new_y):
        dist_thres = 40
        miss_thres = 24

        dist = np.hypot(new_x - self.x, new_y - self.y)
        
        if (dist < dist_thres):
            self.x = new_x
            self.y = new_y
            self.miss_counter = 0

            return(1, self.x, self.y)
        else:
            self.miss_counter = self.miss_counter + 1

            if (self.miss_counter > miss_thres):
                return(0, 0, 0)

            return(1, self.x, self.y)



# Getting Haar Cascade XML
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

    # Capture from videos
    # capture_1 = cv2.VideoCapture(PATH+'out_1.mp4')
    # capture_2 = cv2.VideoCapture(PATH+'out_2.mp4')
    # capture_3 = cv2.VideoCapture(PATH+'out_3.mp4')
    # capture_4 = cv2.VideoCapture(PATH+'out_4.mp4')

    # Capture from cameras    
    capture_1 = cv2.VideoCapture(4)
    capture_2 = cv2.VideoCapture(0)
    capture_3 = cv2.VideoCapture(6)
    capture_4 = cv2.VideoCapture(2)
    
    car = cv2.imread(car_image)
    car = cv2.resize(car, (143, 198))
    
    is_configured = 0
    
    trackerLeftInit = 0
    trackerRightInit = 0
    track_now = 0
    track_left = 0

    car_x = 180
    car_y = 0
    car_w = 144
    car_h = 200

    car_poi1 = (car_x, car_y+car_h-50)
    car_poi2 = (car_x+car_w, car_y+car_h-50)
    car_poi3 = (car_x + car_w//2, car_y + car_h)

    stage = 0

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

        camera4b = camera4[140:410, sideShape[1]-180:]

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
        result[:car.shape[0], camera4b.shape[1]:camera4b.shape[1]+car.shape[1]] = car
        
        if (not is_configured):
            out_noHaar = cv2.VideoWriter(PATH+'out_noHaar.mp4',cv2.VideoWriter_fourcc('M','J','P','G'), 24, (result.shape[1], result.shape[0]))
            out_overall = cv2.VideoWriter(PATH+'out_overall.mp4',cv2.VideoWriter_fourcc('M','J','P','G'), 24, (result.shape[1], result.shape[0]))
            is_configured = 1

        # To save videos without detection
        out_noHaar.write(result)

        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        parkir = parkir_cascade.detectMultiScale(gray, scaleFactor = 1.5, 
                                        minNeighbors = 10, 
                                        minSize=(40,40),
                                        maxSize=(40,40))
        for x, y, w, h in parkir:
            if (((x<180) or (x>324)) or (y>200)):
                result = cv2.rectangle(result, (x,y), (x+w, y+h), (0, 255, 0), 3)
                
        # Init Trackers
        if (track_now == 1):
            tempParkir = sorted(parkir, key = lambda x:x[:][1])
            trackerRight = Tracker(tempParkir[0][0], tempParkir[0][1], tempParkir[0][2], tempParkir[0][3])
            trackerRightInit = 1
            track_now = 0
        if (track_left):
            tempParkir = sorted(parkir, key = lambda x:x[:][0])
            trackerLeft = Tracker(tempParkir[0][0], tempParkir[0][1], tempParkir[0][2], tempParkir[0][3])
            trackerLeftInit = 1
            track_left = 0
        
        # Tracking and Maneuver Recommendation
        if(trackerRightInit):
            for i in range(len(parkir)):
                ok_right, _, _ = trackerRight.track(parkir[i, 0], parkir[i, 1])
            result = cv2.rectangle(result, (trackerRight.x, trackerRight.y), 
                                    (trackerRight.x+trackerRight.w, trackerRight.y+trackerRight.h), (255,0,0), 3)

            if(trackerLeftInit):
                for i in range(len(parkir)):
                    ok_left, _, _ = trackerLeft.track(parkir[i, 0], parkir[i,1])
                result = cv2.rectangle(result, (trackerLeft.x, trackerLeft.y), 
                                        (trackerLeft.x+trackerLeft.w, trackerLeft.y+trackerLeft.h), (255,0,255), 3)

            result = cv2.circle(result, car_poi1, 2, (0,255,0), 3)
            result = cv2.circle(result, car_poi2, 2, (0,255,0), 3)
            result = cv2.circle(result, car_poi3, 2, (0,255,0), 3)


            if(stage == 0):
                result = cv2.putText(result, "Straighten", (10,80), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0), 1)
                result = cv2.putText(result, "Move Forward", (10,100), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0), 1)

            elif(stage ==1):
                result = cv2.putText(result, "Turn Clockwise", (10,80), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0), 1)
                result = cv2.putText(result, "Move Backward", (10,100), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0), 1)
            elif(stage ==2):
                result = cv2.putText(result, "Straighten", (10,80), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0), 1)
                result = cv2.putText(result, "Move Forward", (10,100), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0), 1)
            elif(stage ==3):
                result = cv2.putText(result, "Turn Clockwise", (10,80), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0), 1)
                result = cv2.putText(result, "Move Forward", (10,100), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0), 1)

            else:
                result = cv2.putText(result, "Straighten", (10,80), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0), 1)
                result = cv2.putText(result, "Move Backward", (10,100), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0), 1)

            result = cv2.putText(result, "Stage"+str(stage), (10,60), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0), 1)

            if (car_poi1[1] <= trackerRight.y and stage == 0):
                stage = 1
                dist_thres = 100
                print(stage)
            elif ((np.hypot(car_poi3[0] - trackerRight.x, car_poi3[1] - trackerRight.y) >= 200) and stage == 1):
                stage = 2
                print(stage)
            elif (car_poi1[1] <= trackerRight.y-30 and stage == 2):
                stage = 3
                print(stage)
            elif(trackerLeftInit):
                if ((abs(trackerLeft.y - trackerRight.y)<5)  and stage == 3):
                    stage = 4
                    print(stage)

        # To save video with detection    
        out_overall.write(result)
            
        cv2.imshow("result", result)
        key = cv2.waitKey(1)
        if (key & 0xFF == ord('c')):
            track_now = 1
        elif (key & 0xFF == ord('v')):
            track_left = 1
        elif (key & 0xFF == ord('q')):
            break

if __name__ == "__main__":
    main(save_weights=True)
