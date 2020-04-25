import cv2
import numpy as np 
import yaml
from config import *

name = "camera1"
image_file = "./yaml/{}.jpg".format(name)
camera_file = "./yaml/{}.yaml".format(name)
output = "./yaml/{}_projMat.yaml".format(name)
# horizontal and vertical scaling factors
scale_x = 1.0
scale_y = 1.0
W, H = 640, 480
colors = [(0, 255, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
corners = []
calibDist = 50
canvas_size = (2*W, 2*H)
canvas_center = (W, H)
# -----------------------------------------

shapes = {
    "camera1": leftShape[::-1],
    "camera2": frontShape[::-1],
    "camera3": frontShape[::-1],
    "camera4": leftShape[::-1]
}

dstF = np.float32([
    [shiftWidth + chessboardWidth, shiftHeight],
    [shiftWidth + chessboardWidth + carWidth, shiftHeight],
    [shiftWidth + chessboardWidth, shiftHeight + calibDist],
    [shiftWidth + chessboardWidth + carWidth,  shiftHeight + calibDist]])

dstL = np.float32([
    [ shiftHeight + chessboardHeight,  shiftWidth],
    [ shiftHeight + chessboardHeight + carHeight, shiftWidth],
    [ shiftHeight + chessboardHeight, shiftWidth + calibDist],
    [ shiftHeight + chessboardHeight + carHeight, shiftWidth + calibDist]])

dsts = {"camera1": dstL, "camera2": dstF, "camera3": dstF, "camera4": dstL}


def click(event, x, y, flags, param):
    image, = param
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)
        corners.append((x, y))
    draw_image(image)


def draw_image(image):
    new_image = image.copy()
    for i, point in enumerate(corners):
        cv2.circle(new_image, point, 3, colors[i % 4], -1)

    if len(corners) > 2:
        pts = np.int32(corners).reshape(-1, 2)
        hull = cv2.convexHull(pts)
        mask = np.zeros(image.shape[:2], np.uint8)
        cv2.fillConvexPoly(mask, hull, color=1, lineType=8, shift=0)
        temp = np.zeros_like(image, np.uint8)
        temp[:, :] = [0, 0, 255]
        imB = cv2.bitwise_and(temp, temp, mask=mask)
        cv2.addWeighted(new_image, 1.0, imB, 0.5, 0.0, new_image)

    cv2.imshow("original image", new_image)


def main():
    image = cv2.imread(image_file)

    # Getting matrix parameters from external files
    with open(camera_file, "r") as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)

    K = np.array(data["K"])
    D = np.array(data["D"])
    new_K = K.copy()
    new_K[0, 0] *= scale_x
    new_K[1, 1] *= scale_y

    cv2.namedWindow("original image")
    cv2.setMouseCallback("original image", click, param=[image])
    cv2.imshow("original image", image)
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            return
        elif key == ord("d"):
            if len(corners) > 0:
                corners.pop()
                draw_image(image)
        elif key == 13:
            break

    if len(corners) != 4:
        print("exactly 4 corners are required")
        return

    src = np.float32(corners)
    dst = dsts[name]

    # Generating projection matrix
    print(dst)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, M, (shapes[name][1::-1]))
    
    # warped = cv2.warpPerspective(image, M, (shapes[name][1]*4, shapes[name][0]*4))
    warped = cv2.resize(warped, (shapes[name][1::-1]))
    cv2.imshow("warped", warped)
    cv2.waitKey(0)
    cv2.imwrite(name + "_proj.png", warped)
    print("saving projection matrix to file ...")

    # Exporting Projection Matrix
    with open(camera_file, "a+") as f:
        yaml.safe_dump({"M": M.tolist(), "scale": [scale_x, scale_y]}, f)


if __name__ == "__main__":
    main()