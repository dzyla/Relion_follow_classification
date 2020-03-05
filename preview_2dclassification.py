import mrcfile
import cv2
import easygui
import numpy as np
from os import listdir
from os.path import isfile, join
import re

def plot_classes(classes,classnumber,iter):
    z, x, y = classes.shape
    empty_class = np.zeros((x,y))
    line = []

    if classnumber < 10:
        x_axis = 3
    elif 30 >= classnumber >= 10:
        x_axis = 6
    elif classnumber >= 30:
        x_axis = 10

    if z%x_axis == 0:
        y_axis = int(z / x_axis)
    elif z%x_axis != 0:
        y_axis = int(z / x_axis)+1

    add_extra = int(x_axis*y_axis-z)

    for n, class_ in enumerate(classes):

        if np.average(class_) != 0:
            class_ = (class_ - np.min(class_)) / (np.max(class_) - np.min(class_))

        if n == 0:
            row = class_
        else:
            if len(row) == 0:
                row = class_
            else:
                row = np.concatenate((row,class_),axis=1)
        if (n+1)%x_axis == 0:
            line.append(row)
            row = []

    if add_extra != 0:
        for i in range(0,add_extra):
            row = np.concatenate((row,empty_class), axis=1)
        line.append(row)

    w = 0
    for i in line:
        if w == 0:
            final = i
            w = 1
        else:
            final = np.concatenate((final,i),axis=0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (30, 30)
    fontScale = 1
    fontColor = (1, 1, 1)
    lineType = 2

    cv2.putText(final, 'Iteration {}'.format(iter),
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType)

    win = 1000
    cv2.namedWindow('Real-Time 2D classes', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Real-Time 2D classes' ,win,int(win*y_axis/x_axis))



    cv2.imshow('Real-Time 2D classes', final)

def open_mrcs_file(file_path):
    with mrcfile.open(file_path) as mrc_stack:
        return mrc_stack.data

def real_time_2D():
    import glob
    import os

    folder = easygui.diropenbox()

    last_mrcs = ''

    while True:

        files = glob.glob(folder + '/*.*')
        files.sort(key=os.path.getmtime)
        mrcs_files = []
        for file in files:
            if 'classes.mrcs' in file:
                mrcs_files.append(file)

        pattern2 = '_it\d\d\d_'
        w = re.findall(pattern2, mrcs_files[-1])
        last_iter = (w[0][-4:-1])

        last_mrcs_old = last_mrcs
        last_mrcs = (mrcs_files[-1])

        if last_mrcs != last_mrcs_old:
            mrcs_file_path = join(folder, last_mrcs)
            print(mrcs_file_path)
            mrcs_file = open_mrcs_file(mrcs_file_path)
            z, x, y = mrcs_file.shape

            cv2.destroyAllWindows()

        plot_classes(mrcs_file, z, last_iter)
        if cv2.waitKey(6000) & 0xFF == ord('q'):
            break

        last_mrcs = (mrcs_files[-1])

if __name__ == "__main__":
    real_time_2D()
    cv2.destroyAllWindows()
