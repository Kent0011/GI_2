import cv2
import numpy as np
import random
from tqdm import tqdm
import pandas as pd

#================================================================

SIZE = 200 # 画像サイズ
SIGMA = 5.31 / 2.354
NUMOFDOTS = 5
FILETYPE = '.bmp'
FILENAME = f'/Users/kent/Desktop/GI_2/src/zernike/dots'
PX = 7.5
MLA_F = 14200

#================================================================


def makedot(img: list, i_0: float, j_0: float) -> list:
    
    for i, y in enumerate(img):
        for j,x in enumerate(y):
            img[i][j] = img[i][j] + 255 * np.exp(-((j-j_0)**2+(i-i_0)**2) / (2*(SIGMA**2)))
    
    return img


def makeimage(size, filename):
    
    width, height = size, size
    distance = width/NUMOFDOTS
    
    dots_destination = []
    
    img = np.zeros((height,width), dtype=np.uint8)
    
    df = pd.read_csv("/Users/kent/Desktop/GI_2/src/zernike/doterror.csv", header=None)
    df.columns = ["i","j","x","y"]
    e_x = np.array(df["x"].tolist())
    e_y = np.array(df["y"].tolist())
    
    count=0
    for i in range(NUMOFDOTS):
        for j in range(NUMOFDOTS):
            i_0 = distance/2 + distance*i + e_y[count]*MLA_F/PX
            j_0 = distance/2 + distance*j + e_x[count]*MLA_F/PX
            count+=1
            img = makedot(img, i_0, j_0)
            dots_destination.append((i_0, j_0))
    print(count)

    Name = filename + FILETYPE
    cv2.imwrite(Name, img)
    
    with open(f"{filename}_dots.txt", "w", encoding="utf-8") as file:
        for item in dots_destination:
            file.write(", ".join(map(str, item)) + "\n")




if __name__ == '__main__':
    size = SIZE
    filename = FILENAME
    
    makeimage(size=SIZE, filename=filename)

