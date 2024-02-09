import cv2
import mediapipe as mp
import os
import HandTrackingModule as htm

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

#Leer las imagenes de la carpeta 'fingers_images'
folderPath = "fingers_images"
myList = os.listdir(folderPath)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)

detector = htm.handDetector(detectionCon=0.75)    
fingersId = [4, 8, 12, 16, 20]

while True:
    succes, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = detector.findHands(img)
    lmList = detector.findHandsPositions(img, drawLines=False)

    if len(lmList) != 0:
        fingers = []

        #Comprobamos que el dedo gordo este abierto
        if lmList[fingersId[0]][1] > lmList[fingersId[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)


        for id in range(1,5):
            #Comprobamos que los dedos esten abiertos
            if lmList[fingersId[id]][2] < lmList[fingersId[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        total_fingers = fingers.count(1)    
        #Muestra la imagen en esa posicion
        img[0:200, 0:200] = overlayList[total_fingers]    
        cv2.putText(img, f'Num Dedos: {str(total_fingers)}', (900,70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Hand Count", img)

    cv2.waitKey(1)