import numpy as np
import cv2 
#import pandas as pd
#from PIL import Image, ImageEnhance
#import matplotlib.pyplot as plt
import os 
from datetime import datetime

def imageoptimiser(image):
    img = cv2.imread(image)
    img = cv2.resize(img, (500,500), interpolation = cv2.INTER_AREA)
    img = img + 10
    img.astype(float)
    img = img * 0.8
    img = img.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#   changing contrast and brightness simultaneously
    img= cv2.equalizeHist(img)
    normalizedImg = np.zeros((800, 800))
    img = cv2.normalize(img,  normalizedImg, 0, 255, cv2.NORM_MINMAX)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 15, 2)
#   img = cv2.blur(img, (5, 5))
#   img = img - 40
#   img.astype(float)
#   img = img * 1.7
#   img = img.astype(np.uint8)
    img = img = cv2. GaussianBlur(img, (5,5),0)
#   img = cv2.addWeighted(img, 0.5,img , -0.5, 0.0)
    img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
    img = img[1]
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 15, 2)
    return img


def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        namelist = []
        for line in myDataList:
            entry = line.split(',')
            namelist.append(entry[0])
        if name not in namelist:
            now = datetime.now()
            dstring = now.strftime('%I:%M:%S')
            f.writelines(f"\n{name},{dstring}")



sample = r"C:\Users\CC\Machine_learning_projects\Signals and system OEP\Samples\WhatsApp Image 2022-12-19 at 12.42.46 PM.jpeg"
#sample = input("Enter the path to  the sample image: ")
#sample = str("r") + sample
sample = imageoptimiser(sample)

original_data = r"C:\Users\CC\Machine_learning_projects\Signals and system OEP\registry"

best_score = 0
filename = None 
image = None 
kp1, kp2, mp = None, None, None

for file in os.listdir(original_data):
    fingerprint_img = cv2.imread(original_data + "/" + file  )
    fingerprint_img = imageoptimiser(original_data + "/" + file)
    #fingerprint_img = imageoptimiser(r"C:\Users\CC\Machine_learning_projects\Signals and system OEP\registry\Affan2.jpeg")
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(sample,None)
    keypoints2, descriptors2 = sift.detectAndCompute(fingerprint_img,None)
    matches = cv2.FlannBasedMatcher({'algorithm': 1, 'trees':10},{}).knnMatch(descriptors1,descriptors2,k=2)
    
    match_point = []
    
    # lowes's ratio test
    for p, q in matches:
        if p.distance< 0.7 * q.distance:
            match_point.append(p)
        #else :
         #   match_point.append(q)
            
    keypoints = 0  
    
    if len(keypoints1) < len(keypoints2):
        keypoints = len(keypoints1)
    else:
        keypoints = len(keypoints2)
        
    if len(match_point) / keypoints * 100> best_score:
        best_score = len(match_point) / keypoints * 100
        filename = file
        image = fingerprint_img
        
        kp1, kp2, mp = keypoints1, keypoints2, match_point

print("BEST MATCH: " + filename)   
print("SCORE"+ str(best_score))


result = cv2.drawMatches(sample, kp1 ,image,kp2,mp,None)
result = cv2.resize(result,(500,500))

markAttendance(filename)

cv2.imshow("Result",result)
cv2.waitKey(0)
cv2.destroyAllWindows()
