import cv2
import pickle


if __name__ == '__main__':
    img = cv2.imread("training_chars.png")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(imgBlurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    intValidChars = [ord('0'), ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6'), ord('7'), ord('8'), ord('9'),
                     ord('A'), ord('B'), ord('C'), ord('D'), ord('E'), ord('F'), ord('G'), ord('H'), ord('I'), ord('J'),
                     ord('K'), ord('L'), ord('M'), ord('N'), ord('O'), ord('P'), ord('Q'), ord('R'), ord('S'), ord('T'),
                     ord('U'), ord('V'), ord('W'), ord('X'), ord('Y'), ord('Z')]

    index = dict()
    for contour in contours:
        [intX, intY, intW, intH] = cv2.boundingRect(contour)

        cv2.rectangle(img, (intX, intY), (intX + intW, intY + intH), (0, 0, 255), 2)

        imgROI = thresh[intY:intY + intH, intX:intX + intW]
        imgROIResized = cv2.resize(imgROI, (20, 30))

        cv2.imshow("imgROI", imgROI)
        cv2.imshow("imgROIResized", imgROIResized)
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image', 1000, 600)
        cv2.imshow('image', img)

        intChar = cv2.waitKey(0)

        if intChar in intValidChars:
            imgROIResizedCopy = tuple(imgROIResized.flatten())
            index[imgROIResizedCopy] = intChar

        cv2.destroyAllWindows()

    f = open("letters-and-classifications.pickle", "wb")
    f.write(pickle.dumps(index))
    f.close()

    print("done...indexed %d images" % (len(index)))
    print("Number of images in dictionary", len(index.keys()))
