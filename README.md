# Relion_realtime_preview
Follow your classification and refinement in Relion in the real time!

Simple python script which enables to follow the progress of 2D/3D classification and refinement in Relion.

---Requirements---
* Python 3+ with:
+ matplotlib
+ gemmi
+ mrcfile
+ numpy
+ pandas

---How to run---

Install own python enviroment:
```
python3 -m venv new-env
```
Activate enviroment:
```
source new-env/bin/activate
```
Install packages:
```
pip install numpy matplotlib pandas gemmi mrcfile
```
Run script from command line:
```
python follow_rln_classification.py --i /path/to/classification/folder/ --w wait_time_in_seconds
```
Help can be accesed by:
```
python follow_rln_classification.py --h
usage: follow_rln_classification.py [-h] [--i I] [--w W]

Real-time preview Relion classification output from Class2D, Class3D and
Refine3D jobs, including volume projections, class distributions and estimated
resolution plots

optional arguments:
  -h, --help  show this help message and exit
  --i I       Classification folder path
  --w W       Wait time in seconds between refreshes. Adjust accordingly to
              the expected time of the iteration. If the time is up and there
              no new classification results plots will freeze
```

Double click the .py file and chose the directory where the classification or refinement has already been started. Opened window will update as soon as the iteration is finished.

---Update---

200420: Now single script avaiable for all classification runs, including plotting of class distribution and class estimated resolution. All previous scripts are obsolete. 

~~191010: class2D: now preview also works for the runs which were continued (run_ctXX_itXX...).~~



2D classification preview:
![alt text](https://github.com/dzyla/Relion_realtime_preview/blob/master/cls_projections.png
)


3D classification preview:
![alt text](https://github.com/dzyla/Relion_realtime_preview/blob/master/cls_3d.png
)

Class distribution and estimated resolution plots:
![alt text](https://github.com/dzyla/Relion_realtime_preview/blob/master/class_dist.png
)
![alt text](https://github.com/dzyla/Relion_realtime_preview/blob/master/class_res.png
)
