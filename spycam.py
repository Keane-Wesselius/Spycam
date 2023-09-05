import cv2 as cv
import os


cam = cv.VideoCapture(0)
print(cam.isOpened())
face_classifier = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_face(frame):
	gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
	faces = face_classifier.detectMultiScale(gray_frame, 1.1, 5, minSize=(40, 40))
	for (x, y, w, h) in faces:
		cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)
	return faces


directory = "/home/pi/Python/Webcam"

os.chdir(directory)

filename = "burger2.jpg"

while True and cam.isOpened():
	ret, frame = cam.read()
	faces = detect_face(frame)
	if len(faces) > 0:
		print("we got one boys")
	cv.imshow('Spycam', frame)
	cv.waitKey(1)
	
	
cam.release()
cv.destroyAllWindows()




