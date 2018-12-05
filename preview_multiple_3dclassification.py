import mrcfile
import cv2
import easygui
import numpy as np
from os import listdir
from os.path import isfile, join
import re

def check_mrcs_cv():
    last_mrcs = ''
    folder = easygui.diropenbox()

    while True:
        pattern = re.compile(".*class\d\d\d.mrc")
        pattern2 = '_it\d\d\d_'

        onlyfiles = [f for f in listdir(folder) if isfile(join(folder, f))]
        iter = [file for file in onlyfiles if pattern.match(file)]
        w = re.findall(pattern2, iter[-1])
        last_iter = (w[0][4:-1])
        last_iter_classes = [join(folder, itr) for itr in iter if last_iter in itr]

        last_mrcs_old = last_mrcs
        last_mrcs = (last_iter_classes[-1])

        if last_mrcs != last_mrcs_old:
            print(last_iter_classes[0])

            class_averages = []

            for n, class_ in enumerate(last_iter_classes):
                with mrcfile.open(class_) as mrc_stack:
                    mrcs_file = mrc_stack.data
                    z, x, y = mrc_stack.data.shape

                    average_top = np.zeros((z, y))
                    for i in range(0, z):
                        img = mrcs_file[i, :, :]
                        average_top += img
                    average_top = (average_top - np.min(average_top)) / (np.max(average_top) - np.min(average_top))

                    average_front = np.zeros((z, y))
                    for i in range(0, z):
                        img = mrcs_file[:, i, :]
                        average_front += img
                    average_front = (average_front - np.min(average_front)) / (
                                np.max(average_front) - np.min(average_front))

                    average_side = np.zeros((z, y))
                    for i in range(0, z):
                        img = mrcs_file[:, :, i]
                        average_side += img
                    average_side = (average_side - np.min(average_side)) / (np.max(average_side) - np.min(average_side))

                    average_class = np.concatenate((average_top, average_front, average_side), axis=0)
                    class_averages.append(average_class)

            final_average = np.concatenate(class_averages, axis=1)
            cv2.destroyAllWindows()

        y_axis, x_axis = final_average.shape

        if len(class_averages) == 1:
            win = 300
        else:
            win = 800

        cv2.namedWindow('Real-Time 3D classes, iteration ' + str(last_iter), cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Real-Time 3D classes, iteration ' + str(last_iter), win, int(win * y_axis / x_axis))
        cv2.imshow('Real-Time 3D classes, iteration ' + str(last_iter), final_average)

        if cv2.waitKey(6000) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    check_mrcs_cv()
