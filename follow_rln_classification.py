import argparse
import glob
import os

import mrcfile
import numpy as np
from gemmi import cif
from matplotlib.pyplot import figure, draw, pause
from matplotlib.ticker import MaxNLocator

'''
Relion 2D and 3D classification live preview

requires mrcfile, gemmi and matplotlib. Best to use in own enviroment:

python3 -m venv env
source ./env/bin/activate
pip3 install numpy matplotlib gemmi mrcfile
python3 follow_rln_classification.py --i /path/to/the/classification/ --w 300

Written and maintained by Dawid Zyla, ETH Zurich
'''


def parse_star_model(file_path, loop_name):
    doc = cif.read_file(file_path)

    # block 1 is the per class information
    loop = doc[1].find_loop(loop_name)
    class_data = np.array(loop)

    return class_data


def get_fsc():
    pass


def get_classes(path, model_star_files):
    class_dist_per_run = []
    class_res_per_run = []

    for iter, file in enumerate(model_star_files):
        class_dist_per_run.append(parse_star_model(file, '_rlnClassDistribution'))
        class_res_per_run.append(parse_star_model(file, '_rlnEstimatedResolution'))

    # stack all data together
    class_dist_per_run = np.stack(class_dist_per_run)
    class_res_per_run = np.stack(class_res_per_run)

    # rotate matrix so the there is class(iteration) not iteration(class) and starting from it 0 --> it n
    class_dist_per_run = np.flip(np.rot90(class_dist_per_run), axis=0)
    class_res_per_run = np.flip(np.rot90(class_res_per_run), axis=0)

    # Find the class images (3D) or stack (2D)
    class_files = parse_star_model(model_star_files[-1], '_rlnReferenceImage')

    class_path = []
    for class_name in class_files:
        class_name = os.path.join(path, os.path.basename(class_name))

        # Insert only new classes, in 2D only single file
        if class_name not in class_path:
            class_path.append(class_name)

    n_classes = class_dist_per_run.shape[0]
    iter = class_dist_per_run.shape[1] - 1

    return class_path, n_classes, iter, class_dist_per_run, class_res_per_run


def plot_2dclasses(path, classnumber):

    # open mrcs stack
    classes = mrcfile.open(path[0]).data

    z, x, y = classes.shape
    empty_class = np.zeros((x, y))
    line = []

    if classnumber < 10:
        x_axis = 3
    elif classnumber >= 10:
        x_axis = 12

    if z % x_axis == 0:
        y_axis = int(z / x_axis)
    elif z % x_axis != 0:
        y_axis = int(z / x_axis) + 1

    add_extra = int(x_axis * y_axis - z)

    for n, class_ in enumerate(classes):

        if np.average(class_) != 0:
            try:
                class_ = (class_ - np.min(class_)) / (np.max(class_) - np.min(class_))

            except:
                pass

        if n == 0:
            row = class_
        else:
            if len(row) == 0:
                row = class_
            else:
                row = np.concatenate((row, class_), axis=1)
        if (n + 1) % x_axis == 0:
            line.append(row)
            row = []

    # Fill the rectangle with empty classes
    if add_extra != 0:
        for i in range(0, add_extra):
            row = np.concatenate((row, empty_class), axis=1)
        line.append(row)

    # put lines of images together so the whole rectangle is finished (as a picture)
    w = 0
    for i in line:
        if w == 0:
            final = i
            w = 1
        else:
            final = np.concatenate((final, i), axis=0)

    return final


def plot_3dclasses(files):

    class_averages = []

    for n, class_ in enumerate(files):

        with mrcfile.open(class_) as mrc_stack:
            mrcs_file = mrc_stack.data
            z, x, y = mrc_stack.data.shape

            average_top = np.zeros((z, y))

            # only a slice of the volume is plotted ?
            # for i in range(int(0.45 * z), int(0.55 * z)):

            for i in range(0, z):
                img = mrcs_file[i, :, :]
                average_top += img
            try:
                average_top = (average_top - np.min(average_top)) / (np.max(average_top) - np.min(average_top))
            except:
                pass

            average_front = np.zeros((z, y))

            for i in range(0, z):
                img = mrcs_file[:, i, :]
                average_front += img

            try:
                average_front = (average_front - np.min(average_front)) / (
                        np.max(average_front) - np.min(average_front))
            except:
                pass

            average_side = np.zeros((z, y))

            for i in range(0, z):
                img = mrcs_file[:, :, i]
                average_side += img

            try:
                average_side = (average_side - np.min(average_side)) / (np.max(average_side) - np.min(average_side))
            except:
                pass

            average_class = np.concatenate((average_top, average_front, average_side), axis=0)
            class_averages.append(average_class)

    final_average = np.concatenate(class_averages, axis=1)

    return final_average


def plot_new_stats(axis, data, n_cls):

    for n, class_ in enumerate(data):
        axis.plot(np.arange(0, data.shape[1]), class_.astype(float), label='class {}'.format(n + 1))

    # Add legend if the number of classes is smaller than 20. Otherwise is a mess
    if n_cls < 20:
        axis.legend()

    axis.xaxis.set_major_locator(MaxNLocator(integer=True))


'''------------------------------------------------------------------------------------------------------------------'''


parser = argparse.ArgumentParser(
    description='Real-time preview Relion classification output from Class2D, Class3D and Refine3D jobs, '
                'including volume projections, class distributions and estimated resolution plots')
parser.add_argument('--i', type=str, help='Classification folder path')
parser.add_argument('--w', type=int, default=60,
                    help='Wait time in seconds between refreshes. Adjust accordingly to the expected time of the iteration. '
                         'If the time is up and there no new classification results plots will freeze')
args = parser.parse_args()


old_model_files = ''

# class distribution graph
fg_dist = figure()
ax_dist = fg_dist.gca()

# class resolution graph
fg_res = figure()
ax_res = fg_res.gca()

# class preview graph
fg_class = figure(dpi=150)
ax_class = fg_class.gca()

# add own graph in future

# Start with empty list of old model files
old_model_files = []

while True:

    try:
        # run from the CLI

        path = args.i
        model_files = glob.glob(path + "*model.star")
        model_files.sort(key=os.path.getmtime)

    except:
        # Here for the run without the command line, mostly for debugging

        print('Script should be run with --i /path/to/classification/folder flag')

        path = r'H:\PycharmProjects\untitled1\cryoem\preview_classification\2d'
        model_files = glob.glob(path + "\*model.star")
        model_files.sort(key=os.path.getmtime)

    print('\r', 'Found {} *model.star files'.format(len(model_files)), end='')

    # do the fetching of files only if there is a new file in the folder
    # this is causing that graphs are frozen. Removing it will cause each time the time is up graphs will be updated,
    # causing increase usage of resources.

    if len(old_model_files) != len(model_files):

        (class_paths, n_classes_, iter_, class_dist_, class_res_) = get_classes(path, model_files)

        # 2D classes are in a single file so number of files is different than number of classes
        if len(class_paths) != n_classes_:
            class_image = plot_2dclasses(class_paths, n_classes_)

        elif len(class_paths) == n_classes_:
            class_image = plot_3dclasses(class_paths)

        # clean axis for new plotting, this is how it refreshes plot
        ax_dist.cla()
        ax_res.cla()
        ax_class.cla()

        # plot new data

        for data in np.array([[class_dist_, ax_dist], [class_res_, ax_res]]):
            plot_new_stats(data[1], data[0], n_classes_)

        ax_dist.set_xlabel('Iteration')
        ax_dist.set_ylabel('Class distribution')
        ax_dist.set_title('Class distribution in iteration {}'.format(iter_))
        # fg_dist.tight_layout()

        ax_res.set_xlabel('Iteration')
        ax_res.set_ylabel('Class resolution')
        ax_res.set_title('Class resolution in iteration {}'.format(iter_))
        # ax_res.tight_layout()

        ax_class.imshow(class_image, cmap='gray')
        ax_class.set_axis_off()
        fg_class.tight_layout()
        ax_class.set_title('Class projections in iteration {}'.format(iter_), fontsize=8)

        draw(), pause(args.w)
        old_model_files = model_files
