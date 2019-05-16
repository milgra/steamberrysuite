# motion and face detection for steamberry cafe
# learns new face when space bar is pressed

import os # for file handling
import cv2 # for computer vision
import json # for id/name storing in file
import time # for waiting for camera
import numpy # for converting images to xml data
import imutils # for image rotation

font = cv2.FONT_HERSHEY_SIMPLEX
average = None
last_id = 1
sample_count = 0
id_to_face = { }
has_dataset = False
face_detector = cv2.CascadeClassifier('haarcascade.xml')
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

face_ids = [ ]
face_samples = [ ]

learn_active = False
usbcam_active = True

# read up previous dataset

if not os.path.isdir( 'dataset' ) :

    os.mkdir( "dataset" )
    print( "\nCreating directory 'dataset'" )

if os.path.isfile( 'dataset/facedata.yml' ) :

    face_recognizer.read( 'dataset/facedata.yml' )
    has_dataset = True
    print( "\nFace data loaded" )

if os.path.isfile( 'dataset/idtoname.txt' ) :
    
    file = open( 'dataset/idtoname.txt' , 'r' )
    id_to_face = json.loads( file.read( ) )
    last_id = id_to_face[ "0" ]   # highest id is saved to 0
    file.close( )
    print( "\nName to Id pairs loaded, highest id : " + str( last_id ) )

def detectMotion( image , grayScaleImg ):

    global font
    global average

    # blur the image

    grayScaleImage = cv2.GaussianBlur( grayScaleImg, (21, 21), 0)

    if average is None :

        average = grayScaleImage.copy().astype( "float" )

    else :

        # accumulate the weighted average between the current frame and
	    # previous frames, then compute the difference between the current
	    # frame and running average

        cv2.accumulateWeighted( grayScaleImage, average, 0.5 )
        frameDelta = cv2.absdiff( grayScaleImage , cv2.convertScaleAbs( average ) )

        # threshold the delta image, dilate the thresholded image to fill
	    # in holes, then find contours on thresholded image

        threshold = cv2.threshold( frameDelta, 5, 255, cv2.THRESH_BINARY)[1]
        threshold = cv2.dilate( threshold, None, iterations=2)

        cnts = cv2.findContours(threshold.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        # loop over the contours
        for c in cnts:
        # if the contour is too small, ignore it
            if cv2.contourArea(c) < 5000:
                continue
    
            # compute the bounding box for the contour, draw it on the frame,
            # and update the text
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle( image , (x, y), (x + w, y + h), (0, 155, 0), 2)

def detectFaces( image , grayScaleImage ) :

    global font
    global last_id
    global has_dataset
    global learn_active
    global sample_count
    global face_recognizer
    global face_samples
    global face_ids

    faces = face_detector.detectMultiScale( grayScaleImage , 1.3 , 5 )

    for ( x , y , w , h ) in faces:

            # mark face in image
            cv2.rectangle( image , ( x , y ) , ( x + w , y + h ) , ( 255 , 0 , 0 ) , 2 )     
               
            # train new face in case of train mode
            if learn_active is True :


                faceImg = grayScaleImage[ y : y + h , x : x + w ]

                cv2.putText( image , "Saving face as " + str( last_id ) + "_" + str( sample_count ) , ( x + 5 , y - 5 ) , font , 1 , ( 255 , 255 , 255 ) , 2 )

                # Save the captured face into the datasets folder
                cv2.imwrite( "dataset/User." + str( last_id ) + '.' + str( sample_count ) + ".jpg" , faceImg )
                
                # Sace the captured face in the database
                face_ids.append( last_id )
                face_samples.append( numpy.array( faceImg , 'uint8' ) )

                sample_count += 1

                # Switch off learning
                if ( sample_count == 30 ) :

                    # Save id as unknown in the id data file
                    id_to_face[ str( last_id ) ] = "Unknown"
                    last_id += 1
                    id_to_face[ "0" ] = last_id

                    file = open( 'dataset/idtoname.txt' , 'w' )
                    file.write( json.dumps( id_to_face ) ) 
                    file.close( )

                    sample_count = 0
                    learn_active = False

                    # Train and save model

                    if not has_dataset :
                        face_recognizer.train( face_samples, numpy.array( face_ids ) )
                        has_dataset = True
                    else :
                        face_recognizer.update( face_samples, numpy.array( face_ids ) )

                    # Save the model into trainer/trainer.yml
                    face_recognizer.write('dataset/facedata.yml') 

                    face_samples = [ ]
                    face_ids = [ ]

                    print( "\nModel saved." )

            # detect face if not training
            else :

                id = 0
                confidence = 0

                if has_dataset :

                    id, confidence = face_recognizer.predict( grayScaleImage[ y : y + h , x : x + w ] )

                name = "Guest"
                if confidence > 0 :

                    name = id_to_face[ str( id ) ]

                cv2.putText( image , name + "_" + str( "%.2f" % confidence ) , ( x + 5 , y - 5 ) , font , 1 , ( 255 , 255 , 255 ) , 2 )




def startDetection( ) :

    global learn_active

    cam = None
    raw = None

    if usbcam_active :

        cam = cv2.VideoCapture(0)
        cam.set(3, 640) # set video width
        cam.set(4, 480) # set video height

    else :

        from picamera.array import PiRGBArray
        from picamera import PiCamera

        cam = PiCamera()
        cam.resolution = (640,480)
        cam.framerate = 6
        time.sleep(0.1)
        raw = PiRGBArray(cam)

    while( True ):

        # get image from camera

        if usbcam_active :

            # from usb camera
            result, image = cam.read()

        else :

            # from pi camera
            cam.capture( raw , format = "bgr" , use_video_port = True )
            image = imutils.rotate_bound( raw.array , 90 )    

        # convert to grayscale

        grayScaleImage = cv2.cvtColor( image , cv2.COLOR_BGR2GRAY)

        detectMotion( image , grayScaleImage )
        detectFaces( image , grayScaleImage )

        cv2.imshow( 'camera' , image ) 

        # epty raw rgb array in case of pi camera

        if raw != None :
            raw.truncate( 0 )

        # key handling

        k = cv2.waitKey( 10 ) & 0xff # Press 'ESC' for exiting video

        if k == 27:
            print( "\nExiting" )
            break
        if k == 0x20 and learn_active == False : # Space
            print( "\nStarting training..." )
            learn_active = not learn_active 

startDetection()
