import os
import cv2
import time
import numpy
import imutils
# use picamera if available
import importlib
spam_spec = None

use_picam = True
font = cv2.FONT_HERSHEY_SIMPLEX

cam = None
raw = None

if use_picam :

    print( "using pycam" )

    from picamera.array import PiRGBArray
    from picamera import PiCamera

    cam = PiCamera()
    cam.resolution = (640,480)
    cam.framerate = 6
    time.sleep(0.1)
    raw = PiRGBArray(cam)

else :

    cam = cv2.VideoCapture(0)
    cam.set(3, 640) # set video width
    cam.set(4, 480) # set video height

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
has_dataset = False

sampleCounts={}
faceSamples=[]
ids = []
globalid = 0

learn_mode = False

if os.path.isfile('trainer/trainer.yml') :
    # read up previous dataset
    face_recognizer.read( 'trainer/trainer.yml' )
    has_dataset = True

    # read up names for ids
    

while( True ):

    
    if not use_picam :
        ret, img = cam.read()
    else :
        cam.capture( raw , format = "bgr" , use_video_port = True )
        img = imutils.rotate_bound( raw.array , 90 )

    # detect faces
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    new_faces = False

    for ( x , y , w , h ) in faces:

        # try to detect face

        confidence = 100

        id = globalid

        if has_dataset :

            id, confidence = face_recognizer.predict( gray[ y : y + h , x : x + w ] )

            print( "id " , id , " confidence " , confidence )

        # mark face in image
        cv2.rectangle( img , ( x , y ) , ( x + w , y + h ) , ( 255 , 0 , 0 ) , 2 )     
        cv2.putText( img, "Guest" + str(id), (x+5,y-5), font, 1, (255,255,255), 2)

    if new_faces :
        if has_dataset :
            face_recognizer.update(faceSamples, numpy.array(ids))
        else :
            face_recognizer.train(faceSamples, numpy.array(ids))
        #face_recognizer.write('trainer/trainer.yml')
        has_dataset = True

    if learn_mode:
        cv2.putText(img, "New Faces", (10,25), font, 1, (255,255,255), 2)

    cv2.imshow( 'camera' , img ) 

    raw.truncate(0)

    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
    if k == 0x20: # Space
        learn_mode = not learn_mode 


    time.sleep(0.1)






