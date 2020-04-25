# Parameters for projection (in cm)

carWidth = 160
carHeight = 360
chessboardWidth = 80
chessboardHeight = 110
shiftWidth = 100
shiftHeight = 100
innerShiftWidth = 34
innerShiftHeight = 54
totalWidth = carWidth + 2 * chessboardWidth + 2 * shiftWidth
totalHeight = carHeight + 2 * chessboardHeight + 2 * shiftHeight

x1 = shiftWidth + chessboardWidth + innerShiftWidth
x2 = totalWidth - x1
y1 = shiftHeight + chessboardHeight + innerShiftHeight
y2 = totalHeight - y1

frontShape = (totalWidth, y1)
leftShape = (totalHeight, x1)