import cv2
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt

img = cv2.imread('/Users/kent/Desktop/GI_2/image/testforZERN.bmp')

l = [20, 60, 100, 140, 180]

for i in range(200):
    for j in range(200):
        if (i in l) and (j in l):
            img[i-1:i+2,j-1:j+2] = [255,0,0]
            
plt.imshow(img)
plt.show()