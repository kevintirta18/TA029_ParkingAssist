# Parameters for projection (in cm)

car_W = 160
car_H = 200

overlapChessboard_W = 115
overlapChessboard_H = 115

outerArea_W = 100
outerArea_H = 100

innerDistance_W = 20
innerDistance_H = 8

total_W = car_W + 2 * overlapChessboard_W + 2 * outerArea_W + 2*innerDistance_W
total_H = car_H + overlapChessboard_H + outerArea_H + innerDistance_H

x_left  = outerArea_W + overlapChessboard_W + innerDistance_W
x_right = x_left + car_W
y_up    = 0
y_down  = car_H
x_middle = total_W//2
middle_distance = 50

rearShape = (total_W, total_H-y_down)
sideShape = (total_H, x_left)
