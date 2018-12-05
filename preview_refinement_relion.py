import mrcfile
import cv2
import easygui
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from time import sleep


def open_mrcs(mrcs_file):
    with mrcfile.open(mrcs_file) as mrc:
        if mrc.is_image_stack():
            return mrc.data
        elif len(mrc.data.shape) == 3:
            return mrc.data


def plot_classes(classes, classnumber):
    fig, axes = plt.subplots(4, int(classnumber / 4) + 1, figsize=(10, 10),
                             subplot_kw={'xticks': [], 'yticks': []},
                             gridspec_kw=dict(hspace=0.1, wspace=0.1))
    for i, ax in enumerate(axes.flat):
        if i < len(classes):
            ax.imshow(classes[i],
                      cmap='gray', interpolation='nearest')


def check_mrcs_cv():
    folder = easygui.diropenbox()
    classes_files = []
    last_mrcs = ''
    plot_shown = False

    while True:
        onlyfiles = [f for f in listdir(folder) if isfile(join(folder, f))]

        for file in onlyfiles:
            if '.mrc' in file:
                if 'half1_class' in file:
                    classes_files.append(file)

        last_mrcs_old = last_mrcs
        last_mrcs = (classes_files[-1])

        if last_mrcs != last_mrcs_old:
            mrcs_file = join(folder, last_mrcs)
            print(mrcs_file)

            with mrcfile.open(mrcs_file) as mrc_stack:
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


                average_class = np.concatenate((average_top, average_front,average_side), axis=1)

            y_axis, x_axis = average_class.shape

        win = 800
        cv2.namedWindow('Real-Time 3D Refinement', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Real-Time 3D Refinement', win, int(win * y_axis / x_axis))
        cv2.imshow('Real-Time 3D Refinement', average_class)

        if cv2.waitKey(6000) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    check_mrcs_cv()
