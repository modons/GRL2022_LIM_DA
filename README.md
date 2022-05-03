# GRL2022_LIM_DA

Python source code for the paper:

Hakim, G. J., C. Snyder, S. G. Penny, and M. Newman, 2022: Subseasonal forecast skill improvement from strongly coupled data assimilation with a linear inverse model. Geophysical Research Letters, 49, in press.

CFSR source data can be downloaded here: https://www.ncei.noaa.gov/data/climate-forecast-system/access/reanalysis/time-series

CFSR data should be processed as described in the paper, including EOF files to generate the LIM, and to make observations. Alternatively, convenience files to reproduce results in the paper can be found here: https://www/atmos.uw.edu:/~hakim/GRL2022/

LIM_DA_cycling.py performs the calculations described in the paper, and stores them to a file. Set parameters at the top of the file to perform individual experiments.

LIM_DA_cycling_plot.py plots the results.

LIM_utils_new.py provides utility functions used by these two codes. 

lim_env.yml provides an Anaconda configuration file that may be used to reproduce the python environment used to perform the calculations. The version of all dependencies pertains to those at the time the paper was accepted for publication.

You should first install Anaconda python: https://www.anaconda.com

Then install the environment using:

conda env create -f lim_env.yml

Before running the experiments, activate the environment using:

conda activate lim

---
Greg Hakim
University of Washington
April 2022
