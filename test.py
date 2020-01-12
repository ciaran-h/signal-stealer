import nn
import numpy as np
import cv2
import glob
import os

img_array = []
for filename in sorted(glob.glob('C:\\Users\\Ciaran Hogan\\Desktop\\nn\\*.png'), key=os.path.getmtime):
    print(filename)
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)
 
 
out = cv2.VideoWriter('C:\\Users\\Ciaran Hogan\\Desktop\\test.mp4',cv2.VideoWriter_fourcc(*"mp4v"), 60, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()