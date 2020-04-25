import os
import numpy as np
import yaml
import cv2

# Takes videoCapture input from one camera and do calibration to correct barrel distortion caused by wide-angle lenses
# Calibration use checkerboard pattern

input_camera_num = 2 # Mark each camera with different number
capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Grid Size of checkerboard used = row x column of intersection between white and black square
grid_size = (23, 7)
corners_threshold = 12

# Resolution of the camera
W, H = 640, 480

objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane
grid_points = np.zeros((np.prod(grid_size), 3), np.float32)
grid_points[:, :2] = np.indices(grid_size).T.reshape(-1, 2)

close_windows = False
calibrate = False

while(1):
    _, frame = capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Find intersections between black and white pattern on checkerboard according to grid_size
    is_found, corners = cv2.findChessboardCorners(
            gray,
            grid_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH +
            cv2.CALIB_CB_NORMALIZE_IMAGE +
            cv2.CALIB_CB_FILTER_QUADS
        )
    key = cv2.waitKey(1) & 0xFF

    # Draw each found corners on frame
    if (is_found):
        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), term)
        imgpoints.append(corners.reshape(1, -1, 2))
        objpoints.append(grid_points.reshape(1, -1, 3))
        cv2.drawChessboardCorners(frame, grid_size, corners, is_found)
    cv2.imshow("corners", frame)

    if (key == ord("c")):
        calibrate = True

    elif (key == ord("q")):
        break

    if (calibrate):
        print("\nCalibrating...\n")

        if len(objpoints) < corners_threshold:
            print("Less than threshold corners detected, calibration failed")

        # Find homography matrix
        K = np.zeros((3, 3))
        D = np.zeros((4, 1))
        rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(N_OK)]
        tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(N_OK)]
        calibration_flags = (cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC +
                             cv2.fisheye.CALIB_CHECK_COND +
                             cv2.fisheye.CALIB_FIX_SKEW)
        ret, mtx, dist, rvecs, tvecs = cv2.fisheye.calibrate(
                objpoints,
                imgpoints,
                (W, H),
                K,
                D,
                rvecs,
                tvecs,
                calibration_flags
            )

        print(ret)

        # Store homography matrix in a yaml file
        if ret:
            print(ret)
            data = {"dim": np.array([W, H]).tolist(), "K": K.tolist(), "D": D.tolist()}
            fname = os.path.join("camera" + str(input_camera_num) + ".yaml")
            print(fname)
            with open(fname, "w") as f:
                yaml.safe_dump(data, f)
            print("succesfully saved camera data")
            calibrate=False

        # Test Mapping
        new_K = K.copy()
        new_K[0, 0] *= 1.0
        new_K[1, 1] *= 1.0
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K,
            D,
            np.eye(3),
            new_K,
            (W, H),
            cv2.CV_16SC2
        )

        while(1):
            _, frame = capture.read()
            frame = cv2.remap(
                frame,
                map1,
                map2,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT
            )
            cv2.imshow("FINAL", frame)
            key = cv2.waitKey(1) & 0xFF
            if (key == ord('c')):
                time_stamp = time.time()
                str_time_stamp = datetime.datetime.fromtimestamp(time_stamp).strftime('%d-%m-%Y_%H-%M-%S')
                cv2.imwrite("right-"+str(str_time_stamp)+".jpg", frame)
            if (key == ord('q')):
                break