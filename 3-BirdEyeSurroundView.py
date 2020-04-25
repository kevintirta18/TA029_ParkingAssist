import numpy as np
import cv2
import yaml
import os
from config import *

# Camera1 = Front-Right
# Camera2 = Rear-Right
# Camera3 = Rear-Left
# Camera4 = Front-Left

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


def click_get_pixel(event,x,y,flags,param):
    global refPt, lower, upper
    global is_detected
    if event == cv2.EVENT_LBUTTONDOWN:
        ptx0, pty0 = x,y
        ptx1, pty1 = x-3, y-3
        ptx2, pty2 = x+3, y+3
        is_detected = 1

        pixel = result[pty1:pty2, ptx1:ptx2]
        avg_color_per_row = np.average(pixel, axis=0)
        avg_color = np.average(avg_color_per_row, axis=0)
        lower = avg_color-5
        upper = avg_color+5

def main(save_weights=False):
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

    capture_1 = cv2.VideoCapture(0)
    capture_2 = cv2.VideoCapture(0)
    capture_3 = cv2.VideoCapture(2)
    capture_4 = cv2.VideoCapture(4)

    global counter, lower, upper, result
    while (True):
        # try:
            _, frame_1 = capture_1.read()
            _, frame_2 = capture_2.read()
            _, frame_3 = capture_3.read()
            _, frame_4 = capturqe_4.read()


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
                leftShape
            )
            camera2 = cv2.warpPerspective(
                frame_2,
                matrices[1],
                frontShape
            )
            camera3 = cv2.warpPerspective(
                frame_3,
                matrices[2],
                frontShape
            )
            camera4 = cv2.warpPerspective(
                frame_4,
                matrices[3],
                leftShape
            )

            camera2 = camera2[::-1, ::-1, :]
            camera3 = camera3[::-1, ::-1, :]
            camera1 = np.flip(np.flip(cv2.transpose(camera1)[::-1], 1),0)
            camera4 = np.flip(cv2.transpose(camera4), 0)

            camera1b = camera1[200:570,47:170] #370x123
            camera2b = camera2[90:170, 170:280] #80x110
            camera3b = camera3[100:180,220:340] #80x120
            camera4b = camera4[200:570,60:170] #370x110

            result = np.zeros((camera1b.shape[0], (camera1b.shape[1]+camera2b.shape[1]+camera3b.shape[1]+camera4b.shape[1]+3), 3), dtype=np.uint8)
            result_y = result.shape[0]
            result_x = result.shape[1]
            
            result[:,:110] = camera4b  
            result[result_y-80:, 111:231] = camera3b
            result[result_y-80:, 232:342] = camera2b
            result[:, 343:466] = camera1b

            temp_result = result
            if (cv2.waitKey(1) & 0xFF == ord('q')):
                break    

            cv2.namedWindow("result")
            cv2.setMouseCallback("result", click_get_pixel)
        
            counter = counter+1
            if counter >= 16:

                counter =0

                # create NumPy arrays from the boundaries
                lower = np.array(lower, dtype="uint8")
                upper = np.array(upper, dtype="uint8")

                # find the colors within the specified boundaries and apply
                # the mask
                mask = cv2.inRange(result, lower, upper)
                output = cv2.bitwise_and(result, result, mask=mask)

                ret, thresh = cv2.threshold(mask, 30, 255, 0)
                contours, hierarchy = cv2.findContours(thresh.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)


                if len(contours) != 0:

                    #sorting contours and get the two biggest contour
                    c = sorted(contours, key=cv2.contourArea, reverse=True)
                    c1 = c[0]
                    c2 = c[1]

                    # find the biggest countour (c) by the area

                    rect1 = cv2.minAreaRect(c1)
                    rect2 = cv2.minAreaRect(c2)

                    box1 = cv2.boxPoints(rect1)
                    box2 = cv2.boxPoints(rect2)

                    # mark the points
                    x0,y0 = int(result.shape[1]/2), int(result.shape[0]/2) #starting points (camera)
                    x1,y1 = (box1[0]+box1[2])/2 #line1
                    x2,y2 = (box2[0]+box2[2])/2 #line2

                    is_detected = 1

            if (is_detected):
                # draw image and put text
                result = cv2.line(result,(x0,y0),(x1,y1),(0,255,0),2)
                result = cv2.line(result,(x0,y0),(x2,y2),(0,255,0),2)

                # calculating length
                length1 = np.sqrt((x0-x1)**2+(y0-y1)**2)
                length2 = np.sqrt((x0-x2)**2+(y0-y2)**2)

                cv2.putText(result, str(round(length1)), (x1,y1), 0, 0.5, (0,255,0))
                cv2.putText(result, str(round(length2)), (x2,y2), 0, 0.5, (0,255,0))

            cv2.imshow('result', result)
        # except:
        #     if (cv2.waitKey(1) & 0xFF == ord('q')):
        #         break    
        #     print('pass')
        #     pass
if __name__ == "__main__":
    main(save_weights=True)
