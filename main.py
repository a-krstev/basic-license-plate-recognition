# USAGE
# python main.py --image baza/Pa140044.jpg
# python main.py --image baza/P9190064.jpg
# python main.py --image baza/P9170008.jpg
# python main.py --image baza/P6070058.jpg

import argparse
import cv2
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn import svm
import sys


showSteps = False


def plate_recognition(imagePath):
    img = cv2.imread(imagePath)  # vchituvanje na slikata
    img = cv2.resize(img, (620, 480))  # ova bi pomagalo vo sluchaj da imame slika so visoka rezolucija

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # grey scale

    if showSteps is True:
        cv2.imshow("Grayscale image", gray)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    gray = cv2.bilateralFilter(gray, 11, 17, 17)  # otstranuvanje na shum
    edges = cv2.Canny(gray, 30, 200)  # detekcija na rabovi

    if showSteps is True:
        cv2.imshow("Canny", edges)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  # pronaogjanje na konturite
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]  # gi zadrzhuvame samo najgolemite
    rectangleBox = None  # ovde kje gi chuvame tochkite na pravoagolnata kontura

    for contour in contours:
        perimetar = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimetar, True)

        # ako aproksimiranata kontura ima chetiri tochki, togash pretpostavuvame deka sme ja pronashle tablicata
        if len(approx) == 4:
            rect = cv2.minAreaRect(approx)
            rectangleBox = cv2.boxPoints(rect)
            box = np.int0(approx)  # ova e za iscrtuvanje na konturata
            break

    if rectangleBox is None:
        detected = 0
        # print("No contour detected")
        cv2.imshow("No contour detected", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        detected = 1

    copy = None
    if detected == 1:
        copy = img.copy()  # pravime kopija na originalnata slika bidejkji drawContours funkcijata ja izmenuva
        cv2.drawContours(copy, [box], -1, (0, 255, 0), 3)

        if showSteps is True:
            cv2.imshow("After plate detection", copy)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    return rectangleBox, img, copy


def warp_perspective(plateBounds, image, copy):
    # maskiranje na se osven tablicata
    box = np.int0(plateBounds)
    mask = np.zeros(image.shape[:2], np.uint8)
    newImage = cv2.drawContours(mask, [box], 0, 255, -1)
    newImage = cv2.bitwise_and(copy, copy, mask=mask)

    if showSteps is True:
        cv2.imshow("Detected license plate with mask", newImage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    (x, y) = np.where(mask == 255)
    (topx, topy) = (np.min(x), np.min(y))
    (bottomx, bottomy) = (np.max(x), np.max(y))
    cropped = image[topx:bottomx + 1, topy:bottomy + 1]

    if showSteps is True:
        cv2.imshow("Cropped license plate", cropped)
        cv2.waitKey(0)

    rect = np.zeros((4, 2), dtype="float32")
    s = plateBounds.sum(axis=1)

    # gornata leva tochka go ima najmaliot zbir na x i y koordinatite dodeka dolnata desna tochka najgolemiot
    rect[0] = plateBounds[np.argmin(s)]
    rect[2] = plateBounds[np.argmax(s)]
    diff = np.diff(plateBounds, axis=1)

    # gornata desna tochka ja ima najmalata razlika na x i y koordinatite dodeka dolnata leva tochka najgolemiot
    rect[1] = plateBounds[np.argmin(diff)]
    rect[3] = plateBounds[np.argmax(diff)]
    tl, tr, br, bl = rect

    # shirochinata na novata slika kje bide najgolemata distanca pomegju dolno levata i dolno desnata x koordinata ili
    # gorno levata i gorno desnata x koordinata
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # visinata na novata slika kje bide najgolemata distanca pomegju dolno levata i gorno levata y koordinata ili
    # dolno desnata i gorno desnata y koordinata
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    if showSteps is True:
        cv2.imshow("License plate warped", warped)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return warped


def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0
    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    # return the list of sorted contours and bounding boxes
    return cnts, boundingBoxes


def alphanumerics_recognition(plateImage):
    # vchituvanje na podatocite za SVM modelot
    index = pickle.loads(open("letters-and-classifications.pickle", 'rb').read())
    rawImages = []
    labels = []

    for key in index.keys():
        tmp = list(key)
        rawImages.append(tmp)
        labels.append(int(index[key]))

    rawImages = np.array(rawImages)
    labels = np.array(labels)

    (trainSVM, testSVM, trainSVML, testSVML) = train_test_split(
        rawImages, labels, test_size=0.25, random_state=42)

    print("SVM training...")
    svc = svm.SVC(kernel='linear')
    svc.fit(trainSVM, trainSVML)

    print("SVM testing...")
    acc = svc.score(testSVM, testSVML)
    print("[INFO] raw pixel accuracy: {:.2f}%".format(acc * 100))

    imgTestingNumbers = plateImage.copy()
    gray = cv2.cvtColor(imgTestingNumbers, cv2.COLOR_BGR2GRAY)
    imgBlurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(imgBlurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    characters = []
    contoursSorted, boundingBoxes = sort_contours(contours)  # sortiranje na konturite odlevo nadesno
    for contour in contoursSorted:
        [intX, intY, intW, intH] = cv2.boundingRect(contour)

        # soodnos pomegju povrshinata na konturata i pravoagolnikot koj se opishuva okolu nea
        solidity = cv2.contourArea(contour) / float(intW * intH)
        heightRatio = intH / float(plateImage.shape[0])  # soodnos pomegju visinata na bukvata i tablicata

        if showSteps is True:
            # ova e za da se prikazhuvaat samo detektiranite bukvi ili brojki dodeka gi izminuvame site konturi
            imgTestingNumbers1 = imgTestingNumbers.copy()
            cv2.rectangle(imgTestingNumbers1, (intX, intY), (intX + intW, intY + intH), (0, 0, 255), 2)
            cv2.imshow("Contours detected", imgTestingNumbers1)
            # print(intW * intH, intW, intH, intW / intH, solidity, heightRatio)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # Ovie vrednosti bea odlucheni soodvetno od nasheto podatochno mnozhestvo,
        # sekako deka bi imalo promeni dokolku se promeni istoto
        if 1080 > intW * intH >= 150 and 0.2 <= intW/intH < 1.0 and solidity > 0.2 and 0.6 < heightRatio < 0.95:
            # print(intW * intH, intW, intH, intW / intH, solidity, heightRatio)
            cv2.rectangle(imgTestingNumbers, (intX, intY), (intX + intW, intY + intH), (0, 0, 255), 2)
            imgROI = thresh[intY:intY + intH, intX:intX + intW]
            imgROIResized = cv2.resize(imgROI, (20, 30))

            if showSteps is True:
                cv2.imshow("Alphanumerics detected so far", imgTestingNumbers)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            character = chr(svc.predict([imgROIResized.flatten()]))
            characters.append(character)

    return characters, imgTestingNumbers


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="path to input image")
    args = vars(ap.parse_args())

    file = open("license-plates.txt", "a+")
    rectangleBox, image, imageWithContour = plate_recognition(args["image"])
    if rectangleBox is not None:
        cv2.imshow("Continue processing? Y/N", imageWithContour)
        char = chr(cv2.waitKey(0))
        cv2.destroyAllWindows()

        if char == 'y':
            plateImage = warp_perspective(rectangleBox, image, imageWithContour)
            characters, imgWithAlphanumerics = alphanumerics_recognition(plateImage)
            characters = ''.join(characters)

            print(characters)
            cv2.imshow("Y/N?", imgWithAlphanumerics)  # dali sakate da se zapishe dobieniot rezultat vo datoteka
            char1 = chr(cv2.waitKey(0))
            cv2.destroyAllWindows()

            if char1 == 'y':
                file.write(characters + "\n")
        else:
            print("Bad contour")
    else:
        print("No contour detected")
    file.close()
