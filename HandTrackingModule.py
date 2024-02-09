import cv2
import mediapipe as mp
import time

class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands()
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, drawLines=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        #print(self.results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if drawLines:
                    #Color de los puntos de la mano
                    landmark_drawing_spec = self.mpDraw.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=2)
                    #Color de las lineas de la mano
                    connection_drawing_spec = self.mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2)
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS, landmark_drawing_spec, connection_drawing_spec)
        return img
    
    def findHandsPositions(self, img, handNumbers=0, drawLines=True):
        lmList = []

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNumbers]
            for id, lm in enumerate(myHand.landmark):
                h,w,c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmList.append([id, cx, cy])
                #Poner el punto mas grande en la muñeca o en cualquier dedo, indicando su id
                if id == 0 and drawLines:
                    cv2.circle(img, (cx, cy), 25, (255, 0, 255), cv2.FILLED)

        return lmList



def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720) 

    #Creamos una instancia de la clase handDetector
    detector = handDetector()
    while True:
        succes, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findHandsPositions(img)
        if len(lmList) != 0:
            print(lmList[4])

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        #Muestra los fps
        cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        #Lo que muestra el frame con todo
        cv2.imshow("Hands Detector", img)

        cv2.waitKey(1)

if __name__ == "__main__":
    main()    