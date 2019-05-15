import os
import cv2
import time
import numpy
import imutils
# use picamera if available
import importlib
spam_spec = importlib.util.find_spec( "picamera" )
use_picam = spam_spec is not None
font = cv2.FONT_HERSHEY_SIMPLEX

cam = None

if use_picam :

    from picamera.array import PiRGBArray
    from picamera import PiCamera

    cam = PiCamera()
    cam.resolution = (640,480)
    cam.framerate = 6
    time.sleep(0.1)

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

if os.path.isfile('trainer/trainer.yml') :
    # read up previous dataset
    face_recognizer.read( 'trainer/trainer.yml' )
    has_dataset = True

    # read up previous global id
    imagePaths = [os.path.join("dataset",f) for f in os.listdir("dataset")]     
    for imagePath in imagePaths:
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        if id not in sampleCounts :
            sampleCounts[id] = 0
        else :
            sampleCounts[id] += 1
        print("samplecount for " , id , " : " , sampleCounts[id])
        if id > globalid :
            globalid = id

    globalid += 1
    print( "new globalid " , globalid )

while( True ):

    img = None
    if not use_picam :
        ret, img = cam.read()
    else :
        cam.capture( img , format = "bgr" , use_video_port = True )
        img = cv2.flip(img, -1) # flip video image vertically

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

        if confidence > 50 :

            id = globalid
            print( "storing new face" )
            new_faces = True

            newface = gray[ y : y + h , x : x + w ]
            rotated = imutils.rotate(newface, 0 )

            for i in range(30) :
                rotated = imutils.rotate(rotated, 0 )
                img_numpy = numpy.array( rotated , 'uint8' )
                faceSamples.append( img_numpy )
                ids.append( id )    
                cv2.imwrite( "dataset/User." + str( id ) + "." + str(i) + ".jpg" , rotated )
            
            globalid += 1
            sampleCounts[id] = 30

        else :

            if sampleCounts[id] < 40 :
                print( "needs more samples (" , sampleCounts[id] , ")" )
                new_faces = True
                img_numpy = numpy.array( gray , 'uint8' )
                faceSamples.append( img_numpy[ y : y + h , x : x + w ] )
                ids.append( id )
                cv2.imwrite( "dataset/User." + str( id ) + "." + str(sampleCounts[id]) +  ".jpg" , gray [ y : y + h , x : x + w ] )
                sampleCounts[id] += 1

        # mark face in image
        cv2.rectangle( img , ( x , y ) , ( x + w , y + h ) , ( 255 , 0 , 0 ) , 2 )     
        cv2.putText(img, "Guest" + str(id), (x+5,y-5), font, 1, (255,255,255), 2)

    if new_faces :
        
        if has_dataset :
            face_recognizer.update(faceSamples, numpy.array(ids))
        else :
            face_recognizer.train(faceSamples, numpy.array(ids))
        #face_recognizer.write('trainer/trainer.yml')
        has_dataset = True

    cv2.imshow( 'camera' , img ) 

    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break

    #time.sleep(0.5)






