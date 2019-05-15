# import the necessary packages
import argparse
import warnings
import datetime
import imutils
import json
import time
import cv2

 # import the necessary packages
import uuid
import os
 
class TempImage:
	def __init__(self, basePath="./", ext=".jpg"):
		# construct the file path
		self.path = "{base_path}/{rand}{ext}".format(base_path=basePath,
			rand=str(uuid.uuid4()), ext=ext)
 
	def cleanup(self):
		# remove the file
		os.remove(self.path)

# filter warnings, load the configuration and initialize the Dropbox
# client
warnings.filterwarnings("ignore")
conf = json.load(open("motiondetectorconf.json"))
client = None

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
 
# allow the camera to warmup, then initialize the average frame, last
# uploaded timestamp, and frame motion counter
print("[INFO] warming up...")
time.sleep(conf["camera_warmup_time"])
avg = None
lastUploaded = datetime.datetime.now()
motionCounter = 0

# capture frames from the camera
while( True ):

    img = None
    if not use_picam :
        ret, img = cam.read()
    else :
        cam.capture( img , format = "bgr" , use_video_port = True )
        img = cv2.flip(img, -1) # flip video image vertically

	# grab the raw NumPy array representing the image and initialize
	# the timestamp and occupied/unoccupied text
    timestamp = datetime.datetime.now()
    text = "Unoccupied"
 
    img = imutils.resize(img, width=500)
	# resize the frame, convert it to grayscale, and blur it
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
 
	# if the average frame is None, initialize it
    if avg is None:
        print("[INFO] starting background model...")
        avg = gray.copy().astype("float")
        continue
 
	# accumulate the weighted average between the current frame and
	# previous frames, then compute the difference between the current
	# frame and running average
    cv2.accumulateWeighted(gray, avg, 0.5)
    frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))
    # threshold the delta image, dilate the thresholded image to fill
	# in holes, then find contours on thresholded image
    thresh = cv2.threshold(frameDelta, conf["delta_thresh"], 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
 
	# loop over the contours
    for c in cnts:
	# if the contour is too small, ignore it
        if cv2.contourArea(c) < conf["min_area"]:
            continue
 
    	# compute the bounding box for the contour, draw it on the frame,
		# and update the text
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = "Occupied"
 
	# draw the text and timestamp on the frame
    ts = timestamp.strftime("%A %d %B %Y %I:%M:%S%p")
    cv2.putText(img, "Room Status: {}".format(text), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(img, ts, (10, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,	0.35, (0, 0, 255), 1)
        # check to see if the room is occupied
    if text == "Occupied":
		# check to see if enough time has passed between uploads
        if (timestamp - lastUploaded).seconds >= conf["min_upload_seconds"]:
			# increment the motion counter
            motionCounter += 1
 
			# check to see if the number of frames with consistent motion is
			# high enough
            if motionCounter >= conf["min_motion_frames"]:
				# check to see if dropbox sohuld be used
                if conf["use_dropbox"]:
					# write the image to temporary file
                    t = TempImage()
                    cv2.imwrite(t.path, img)
 
					# upload the image to Dropbox and cleanup the tempory image
                    print("[UPLOAD] {}".format(ts))
                    path = "/{base_path}/{timestamp}.jpg".format(
					    base_path=conf["dropbox_base_path"], timestamp=ts)
                    #client.files_upload(open(t.path, "rb").read(), path)
                    #t.cleanup()
 
				# update the last uploaded timestamp and reset the motion
				# counter
                lastUploaded = timestamp
                motionCounter = 0
 
	# otherwise, the room is not occupied
    else:
        motionCounter = 0

    # check to see if the frames should be displayed to screen
    if conf["show_video"]:
		# display the security feed
        cv2.imshow("Security Feed", img)
        key = cv2.waitKey(1) & 0xFF
 
		# if the `q` key is pressed, break from the lop
        if key == ord("q"):
            break
 