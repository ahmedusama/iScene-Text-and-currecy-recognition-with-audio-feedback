
# coding: utf-8

# In[1]:


from PIL import Image
import pytesseract
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import glob 
import pyttsx3
import multiprocessing
import time


# In[2]:


engine = pyttsx3.init()
cap = cv2.VideoCapture(0)
imeges=glob.glob('*.jpg') 
count = 0
success = True
good_matches=[]
MIN_MATCH_COUNT = 20
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
x=1
sift = cv2.xfeatures2d.SIFT_create()


# In[3]:


def OCR (image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
    gray = cv2.threshold(gray,0,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    #gray = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    sobelx64f = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    abs_sobel64f = np.absolute(sobelx64f)
    sobel_8u = np.uint8(abs_sobel64f)


    kernal = np.ones((8,8),np.uint8)
    erode = cv2.erode(sobel_8u,kernal,iterations = 1)
    dilation = cv2.dilate(erode,kernal,iterations = 2)
    while(1):
    
     cv2.imshow('image',dilation)
     if cv2.waitKey(30)==ord('t'):
        cv2.destroyAllWindows()

        break
    #plt.figure(figsize=(20,10))
    #plt.imshow(dilation,cmap='gray')
    #plt.show()
    contours = cv2.findContours(dilation,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x,y,w,h = cv2.boundingRect(cnt)
    img = cv2.rectangle(image,(int(x*0.3),int(y*0.3)),(int((x+w)*1.2),int((y+h)*1.2)),(0,0,255),6)
    while(1):
    
     cv2.imshow('hi',image)
     if cv2.waitKey(30)==ord('t'):
        cv2.destroyAllWindows()

        break
    x2 = int(x*0.3)
    y2 = int(y*0.3)
    e1 = int((x+w)*1.2)
    e2 = int((y+h)*1.2)
    crop_img = img[y2:e2, x2:e1] #CROP
    #image = cv2.imread("/home/eslam/Downloads/figure-65.png")
    # convert the image to grayscale and flip the foreground
    # and background to ensure foreground is now "white" and
    # the background is "black"
    while(1):
    
     cv2.imshow('image',crop_img)
     if cv2.waitKey(30)==ord('t'):
        cv2.destroyAllWindows()

        break
    gray1 = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.bitwise_not(gray1)

    # threshold the image, setting all foreground pixels to
    # 255 and all background pixels to 0
    thresh1 = cv2.threshold(gray1, 0, 255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    #thresh1 = cv2.adaptiveThreshold(gray1,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)

    # grab the (x, y) coordinates of all pixel values that
    # are greater than zero, then use these coordinates to
    # compute a rotated bounding box that contains all
    # coordinates
    coords = np.column_stack(np.where(thresh1 > 0))
    angle = cv2.minAreaRect(coords)[-1]

    # the `cv2.minAreaRect` function returns values in the
    # range [-90, 0); as the rectangle rotates clockwise the
    # returned angle trends to 0 -- in this special case we
    # need to add 90 degrees to the angle
    if angle < -45:
        angle = -(90 + angle)

    # otherwise, just take the inverse of the angle to make
    # it positive
    else:
        angle = -angle

    # rotate the image to deskew it
    (h1, w1) = crop_img.shape[:2]
    center = (w1 // 2, h1 // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(crop_img, M, (w1, h1),
    flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    rotated=cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    #rotated=cv2.bitwise_not(rotated)
    #rotated = cv2.threshold(rotated,40,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    rotated = cv2.adaptiveThreshold(rotated,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,15,7)
   

    while(1):
    
     cv2.imshow('image',rotated)
     if cv2.waitKey(30)==ord('t'):
        cv2.destroyAllWindows()

        break
    # draw the correction angle on the image so we can validate it
    #cv2.putText(rotated, "Angle: {:.2f} degrees".format(angle),
    #(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    filename = "{}.png".format(os.getpid())
    cv2.imwrite(filename,rotated)
    text = pytesseract.image_to_string(Image.open(filename))
    #os.remove(filename)
    print(text)
    return text


# In[4]:


wait=0
while 1 :
   
    for i in imeges:
        ret,img1 = cap.read()
        img1=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
   # queryImage
        img2 = cv2.imread(i,0)

  # Initiate SIFT detector

  # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1,None)
        kp2, des2 = sift.detectAndCompute(img2,None)

        matches = flann.knnMatch(des1,des2,k=2)

  # store all the good matches as per Lowe's ratio test.
        good = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)
        if len(good)>MIN_MATCH_COUNT:
            print (i)
            if ((i == "5-back.jpg")| (i =="5-front.jpg")):
                engine.say("kaamssaa geneeeh")
                engine.runAndWait()
            if ((i=="10-back.jpg") |(i=="10-front.jpg")):
                engine.say("aashraa geneeeh")
                engine.runAndWait()
            if ((i=="20-back.jpg" )|(i=="20-front.jpg")):
                engine.say("eeshreen geneeeh")
                engine.runAndWait()
            if ((i=='50-back.jpg') |(i=='50-front.jpg')):
                engine.say("khaaammseen geneeeh")
                engine.runAndWait()
            if ((i=='100-back.jpg') |(i=='100-front.jpg')):
                engine.say("meeeet geneeeh")
                engine.runAndWait()
            if ((i=='200-back.jpg' )|(i=='200-front.jpg')):
                engine.say("meetteeen geneeeh")
                engine.runAndWait()
            src_ptss = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            matchesMask = mask.ravel().tolist()

            h,w = img1.shape
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            dst = cv2.perspectiveTransform(pts,M)
            img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
        else:
            print ("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
            matchesMask = None
        draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

        img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
        good_matches.append(good)

        cv2.imshow('Matches ',img3)
        if cv2.waitKey(30)==ord('q'):
            cv2.destroyAllWindows()
            cap.release()
            x=0
            break
        elif cv2.waitKey(30)==ord('s'):
                cv2.destroyAllWindows()
                while 1:
                 ret,img1 = cap.read()
                 cv2.imshow('image',img1)
                 if cv2.waitKey(30)==ord('t'):
                    break
                text=OCR(img1)
                engine.say(text)
                engine.runAndWait()
                cv2.destroyAllWindows()
                


