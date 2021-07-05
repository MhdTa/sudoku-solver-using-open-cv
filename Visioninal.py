import cv2
import imutils
import numpy as np
from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border
import joblib
from sudoku import Sudoku
from compare import predict_digit, init_hog_descripter, hog_compute
from hi import getLines
import time

import traceback
import logging


def getDigit(cell):
    # cv2.imshow('fdsg',cell)
    # cv2.waitKey(0)
    thresh = cv2.threshold(cell, 0, 255,
                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    thresh = clear_border(thresh)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    if len(cnts) == 0:
        return None

    c = max(cnts, key=cv2.contourArea)
    mask = np.zeros(thresh.shape, dtype="uint8")
    cv2.drawContours(mask, [c], -1, 255, -1)

    (h, w) = thresh.shape
    percentFilled = cv2.countNonZero(mask) / float(w * h)

    if percentFilled < 0.03:
        return None

    digit = cv2.bitwise_and(thresh, thresh, mask=mask)

    # cv2.imshow("Digit", digit)
    # cv2.waitKey(0)

    return digit


hog = init_hog_descripter()
svm = cv2.ml.SVM_load('svm_train-test.xml')
video_url = 'http://192.168.43.1:8080/video'
cap = cv2.VideoCapture(video_url)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)
flag = False
while True:
    ret = True
    ret, frame = cap.read()

    # frame = cv2.imread('C:\\Users\Omar\Desktop\Vision\sudoku1.jpg')
    # frame = cv2.imread('rr.jpg')
    # print(frame.shape)
    pressed_key = cv2.waitKey(1)
    if ret:
        try:
            t = time.time()
            image = frame
            # cv2.imshow('1', frame)
            # continue
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (3, 3), 3)

            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 2)
            thresh = cv2.bitwise_not(thresh)
            # cv2.imshow("Puzzle Thresh", thresh)
            # cv2.waitKey(0)

            cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)

            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

            puzzleCnt = None
            for c in cnts:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                if len(approx) == 4:
                    puzzleCnt = approx
                    break

            output = image.copy()
            cv2.drawContours(output, [puzzleCnt], -1, (0, 255, 0), 2)
            cv2.imshow("Puzzle Outline", output)
            # cv2.waitKey(0)

            puzzle_img = four_point_transform(image, puzzleCnt.reshape(4, 2))
            if puzzle_img is None:
                continue
            warped = four_point_transform(gray, puzzleCnt.reshape(4, 2))
            if warped is None:
                continue
            if not flag:
                l = len(getLines(warped))
                print(l)
                if l > 9:

                    board = np.zeros((9, 9), dtype="int")

                    stepX = warped.shape[1] // 9
                    stepY = warped.shape[0] // 9

                    cellLocs = []
                    t = time.time()
                    for y in range(0, 9):
                        row = []
                        for x in range(0, 9):
                            startX = x * stepX
                            startY = y * stepY
                            endX = (x + 1) * stepX
                            endY = (y + 1) * stepY
                            row.append((startX, startY, endX, endY))

                            cell = warped[startY:endY, startX:endX]
                            digit = getDigit(cell)
                            if digit is None:
                                # board[y, x] = 0
                                continue
                            roi = cv2.resize(digit, (32, 32), interpolation=cv2.INTER_AREA)
                            # print(time.time() - t)

                            # print(roi.shape)
                            hist = hog_compute(hog, roi)
                            # print(hist.shape)
                            hist = np.reshape(hist, (-1))
                            hist = np.matrix([hist]).astype(np.float32)
                            _, nbr = svm.predict(hist)
                            # nbr = clf.predict(np.array([hist], 'float64'))
                            # print(nbr)
                            board[y, x] = nbr[0]
                            # print(time.time() - t)
                        cellLocs.append(row)
                    # cv2.imshow("Sudoku Result", puzzle_img)
                    # if board == np.zeros((9, 9), dtype="int"):
                    #     print('no')
                    #     continue
                    puzzle = Sudoku(3, 3, board=board.tolist())
                    # puzzle.show()
                    # solve the Sudoku puzzle
                    # print("[INFO] solving Sudoku puzzle...")
                    if (puzzle.validate()):
                        fainal = puzzle
                        flag = True
                        print('solving')
                        solution = puzzle.solve()
                        print(time.time() - t)

                        solution.show_full()
                        for (cellRow, boardRow) in zip(cellLocs, solution.board):
                            # loop over individual cell in the row
                            for (box, digit) in zip(cellRow, boardRow):
                                # unpack the cell coordinates
                                startX, startY, endX, endY = box
                                # compute the coordinates of where the digit will be drawn
                                # on the output puzzle image
                                textX = int((endX - startX) * 0.33)
                                textY = int((endY - startY) * -0.2)
                                textX += startX
                                textY += endY
                                # draw the result digit on the Sudoku puzzle image
                                cv2.putText(puzzle_img, str(digit), (textX, textY),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                        cv2.imshow("Sudoku Result", puzzle_img)
                    else:
                        for (cellRow, boardRow) in zip(cellLocs, board):
                            # loop over individual cell in the row
                            for (box, digit) in zip(cellRow, boardRow):
                                # unpack the cell coordinates
                                startX, startY, endX, endY = box
                                # compute the coordinates of where the digit will be drawn
                                # on the output puzzle image
                                textX = int((endX - startX) * 0.33)
                                textY = int((endY - startY) * -0.2)
                                textX += startX
                                textY += endY
                                # draw the result digit on the Sudoku puzzle image
                                if digit == 0:
                                    digit = ' '
                                cv2.putText(puzzle_img, str(digit), (textX, textY),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                        cv2.imshow("Sudoku Result", puzzle_img)
            else:
                for (cellRow, boardRow) in zip(cellLocs, fainal):
                    # loop over individual cell in the row
                    for (box, digit) in zip(cellRow, boardRow):
                        # unpack the cell coordinates
                        startX, startY, endX, endY = box
                        # compute the coordinates of where the digit will be drawn
                        # on the output puzzle image
                        textX = int((endX - startX) * 0.33)
                        textY = int((endY - startY) * -0.2)
                        textX += startX
                        textY += endY
                        # draw the result digit on the Sudoku puzzle image
                        if digit == 0:
                            digit = ' '
                        cv2.putText(puzzle_img, str(digit), (textX, textY),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                cv2.imshow("Sudoku Result", puzzle_img)
            # show the output image
        except Exception as e:
            logging.error(traceback.format_exc())

        # cv2.imshow("asd" , image)
        # if cv2.waitKey(1) == 27:
        #     break
        if cv2.waitKey(1) == ord('e'):
            print('hhhhhhhhhhhhhhhhhhhhhhhhhhh4')
            flag = False
