from keras.preprocessing.image import img_to_array, list_pictures, load_img, array_to_img
import cv2
import random
import numpy as np
from PIL import Image
import os
import glob

sendsize = 700
sendimg = int(sendsize/3)
row = 256
col = 256
noise_amount = 0.12

min_table = 50
max_table = 205
diff_table = max_table - min_table

LUT_HC = np.arange(256, dtype='uint8')
LUT_LC = np.arange(256, dtype='uint8')

# ハイコントラストLUT作成
for i in range(0, min_table):
    LUT_HC[i] = 0
for i in range(min_table, max_table):
    LUT_HC[i] = 255 * (i - min_table) / diff_table
for i in range(max_table, 255):
    LUT_HC[i] = 255

# ローコントラストLUT作成
for i in range(256):
    LUT_LC[i] = min_table + i * (diff_table) / 255


def makeresize(file_name, extension):
    img = cv2.imread(file_name+extension)
    madeimg = cv2.resize(img, dsize=(row, col))

    cv2.imwrite(file_name+"_resize.bmp", madeimg)


def makenoise(file_name, extension):
    img = cv2.imread(file_name+extension)
    madeimg = cv2.resize(img, dsize=(row, col))

    ind = list(range(sendimg+1, row*col-sendimg*2, sendimg))
    randomindex = random.sample(range(0, len(ind)), int(len(ind)*noise_amount))
    ind = [ind[i] for i in randomindex]

    for i in ind:
        # print(i)
        for j in range(sendimg):
            madeimg[(i+j)//row][(i+j) % col] = [0, 0, 0]
    cv2.imwrite(file_name+"_noise.bmp", madeimg)


def makerotate(file_name, extension):
    img = cv2.imread(file_name+extension)
    madeimg = cv2.resize(img, dsize=(row, col))
    p90img = madeimg.transpose(1, 0, 2)[:, ::-1]  # +90度
    m90img = madeimg.transpose(1, 0, 2)[::-1]  # -90度

    cv2.imwrite(file_name+"_p90.bmp", p90img)
    cv2.imwrite(file_name+"_m90.bmp", m90img)


def makecontrast(file_name, extension):
    img = cv2.imread(file_name+extension)
    madeimg = cv2.resize(img, dsize=(row, col))

    high_cont_img = cv2.LUT(madeimg, LUT_HC)
    low_cont_img = cv2.LUT(madeimg, LUT_LC)
    cv2.imwrite(file_name+"_highcont.bmp", high_cont_img)
    cv2.imwrite(file_name+"_lowcont.bmp", low_cont_img)


if __name__ == '__main__':
    files = glob.glob('./exp/*/*/*/*.JPG')

    for f in files:
        print(f)
        ftitle, text = os.path.splitext(f)
        makeresize(ftitle, text)
        makenoise(ftitle, text)
        makerotate(ftitle, text)
        makecontrast(ftitle, text)
        os.remove(f)
