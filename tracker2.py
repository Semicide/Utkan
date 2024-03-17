import numpy as np
import cv2

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cap = cv2.VideoCapture(0)

out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 15., (1080,720))

while True:
    ret, frame = cap.read()

    frame = cv2.resize(frame, (1080, 720))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8))

    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

    for (xA, yA, xB, yB) in boxes:
        cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 0, 255), 2)

    out.write(frame.astype('uint8'))
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break





cap.release()
out.release()
cv2.destroyAllWindows()
cv2.waitKey(1)
