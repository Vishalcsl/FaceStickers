import cv2
import numpy as np 
import pandas as pd

eye_cascade = cv2.CascadeClassifier('../Train/third-party/frontalEyes35x16.xml')
nose_cascade = cv2.CascadeClassifier('../Train/third-party/haarcascade_mcs_nose.xml')
face_cascade = cv2.CascadeClassifier('../Train/third-party/haarcascade_frontalface_alt.xml')
#Loading  the images
img = cv2.imread('../Test/Before.png')
imgGlass = cv2.imread('../Train/glasses.jpg',-1)
imgMustache = cv2.imread('../Train/moustacheAg.png',-1) # -1 specifies to load all the layers of image
imgGlassGray = cv2.cvtColor(imgGlass,cv2.COLOR_BGR2GRAY)
#ret, orig_mask_glass = cv2.threshold(imgGlassGray,10,255,cv2.THRESH_BINARY)
#create mask for mustache and eye
orig_mask = imgMustache[:,:,3]
orig_mask_glass = imgGlass[:,:,3]
#Create the Inverted mask for the mustache
orig_mask_inv = cv2.bitwise_not(orig_mask)
orig_mask_inv_glass = cv2.bitwise_not(orig_mask_glass)
#Convert mustache image to BGR and save the  original image size
imgMustache = imgMustache[:,:,0:3]
origMustacheHeight, origMustacheWidth = imgMustache.shape[:2]
imgGlass = imgGlass[:,:,0:3]
origGlassHeight, origGlassWidth = imgGlass.shape[:2]
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(
	gray,
	scaleFactor = 1.1,
	minNeighbors = 5,
	minSize = (30,30),
	flags = cv2.CASCADE_SCALE_IMAGE
	)
print(len(faces))
#Iterating over each face found in the image
for (x,y,w,h) in faces:
	#face = cv2.rectangle(img, (x,y),(x+w,y+h),(255,0,0),2)

	#calculating region of interest
	roi_gray = gray[y:y+h, x:x+w]
	roi_color = img[y:y+h, x:x+w]
	cv2.imshow("ROI_GRAY",roi_gray)
	cv2.imshow("ROI_COLOR",roi_color)

	#Detect a nose within the region bounded by each face (the ROI)
	noses = nose_cascade.detectMultiScale(roi_gray)
	print(len(noses),noses.shape)
	cnt=0

	for nose in noses:
		if cnt==0:

			nx,ny,nw,nh = nose
			offset = ny/4
			#cv2.rectangle(roi_color,(nx,ny),(nx+nw,ny+nh),(255,0,0),2)

			#The mustache should be three times the width of the nose
			mustacheWidth = 3 * nw
			mustacheHeight = mustacheWidth * origMustacheHeight / origMustacheWidth

			#Centre the mustache on the bottom of the node
			x1 = int(nx - (mustacheWidth/4))
			x2 = int(nx + nw + (mustacheWidth/4))
			y1 = int(ny - (mustacheHeight/6)+offset)
			y2 = int(ny + nh + (mustacheHeight/6)+offset)

			#Check for clipping
			if x1 < 0:
				x1 = 0
			if y1 < 0:
				y1 = 0
			if x2 > w:
				x2 = w
			if y2 > h:
				y2 = h

			#Recalculate the width and height of the mustache
			mustacheWidth = int(x2 - x1)
			mustacheHeight = int(y2 - y1)

			#Resize the image and the masks to the size calculated above
			mustache = cv2.resize(imgMustache, (mustacheWidth,mustacheHeight), interpolation = cv2.INTER_AREA)
			mask = cv2.resize(orig_mask, (mustacheWidth,mustacheHeight), interpolation = cv2.INTER_AREA)
			mask_inv = cv2.resize(orig_mask_inv, (mustacheWidth,mustacheHeight), interpolation = cv2.INTER_AREA)

			#Take ROI for mustache from background equal to size of mustache image
			roi = roi_color[y1:y2, x1:x2]

			#roi_bg contains the original image only where the mustache is not in
			#the region that is the size of the mustache
			roi_bg = cv2.bitwise_and(roi,roi,mask=mask_inv)

			#roi_fg contains the image of the mustache only where the mustache is
			roi_fg = cv2.bitwise_and(mustache,mustache,mask=mask)

			#join roi_bg and roi_fg
			dst = cv2.add(roi_bg,roi_fg)

			#replace the joined image, saved to dst back over the original image 
			roi_color[y1:y2, x1:x2] = dst

		cnt+=1

	eyes = eye_cascade.detectMultiScale(roi_gray)
	print(len(eyes),eyes.shape)

	for eye in eyes:
		ex,ey,ew,eh = eye
		#cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,0,0),2)

		glassWidth = 1.4 * ew
		glassHeight = glassWidth * origGlassHeight / origGlassWidth

		x1 = int(ex - (glassWidth/4))
		x2 = int(ex + ew +(glassWidth/4))
		y1 = int(ey - (glassHeight/2))
		y2 = int(ey + eh + (glassHeight/2))

		if x1<0:
			x1=0
		if x2>w:
			x2=w
		if y1<0:
			y1=0
		if y2>h:
			y2=h

		glassWidth = int(x2 - x1)
		glassHeight = int(y2 - y1)

		glasses = cv2.resize(imgGlass, (glassWidth, glassHeight), interpolation = cv2.INTER_AREA)
		mask_glass = cv2.resize(orig_mask_glass, (glassWidth, glassHeight), interpolation = cv2.INTER_AREA)
		mask_glass_inv = cv2.resize(orig_mask_inv_glass, (glassWidth, glassHeight), interpolation = cv2.INTER_AREA)

		roi = roi_color[y1:y2, x1:x2]

		roi_bg = cv2.bitwise_and(roi,roi,mask=mask_glass_inv)
		roi_fg = cv2.bitwise_and(glasses,glasses,mask=mask_glass)

		dst = cv2.add(roi_bg,roi_fg)

		roi_color[y1:y2, x1:x2] = dst


#Displaying the resulting frame
cv2.imshow("Resulting Image", img)
cv2.imwrite('../Test/After.png',img)
#storing the pixels
img = img.flatten()
img = img.reshape((-1,3))
print(img.shape)
df = pd.DataFrame({'Channel 1':img[1:,0],
	'Channel 2':img[1:,1],
	'Channel 3':img[1:,2]})
df.to_csv('../Test/output.csv',index=False)

#closing  the file
cv2.waitKey(0)
cv2.destroyAllWindows()