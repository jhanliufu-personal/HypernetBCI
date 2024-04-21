import matplotlib.pyplot as plt
from braindecode.datasets import MOABBDataset
from numpy import multiply
from braindecode.preprocessing import (Preprocessor,
                                       exponential_moving_standardize,
                                       preprocess)
from braindecode.preprocessing import create_windows_from_events
import torch
from braindecode.models import ShallowFBCSPNet
from braindecode.util import set_random_seeds
from skorch.callbacks import LRScheduler
from skorch.helper import predefined_split
from braindecode import EEGClassifier
# from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import os
import pickle
from matplotlib.lines import Line2D
# from braindecode.visualization import plot_confusion_matrix

from braindecode.datasets import BaseConcatDataset
from braindecode.datasets.base import EEGWindowsDataset
from braindecode.preprocessing.windowers import _create_windows_from_events
import numpy as np
import mne
import random

# This will download data to default path, ~mne_data.
# Need to figure out how to specify download path
# BNCI2014_001 has 9 subjects
BNCI2014_001_dataset = MOABBDataset(dataset_name="BNCI2014_001", subject_ids=list(range(1, 10)))
# Schirrmeister2017 has 14 subjects
Schirrmeister2017_dataset = MOABBDataset(dataset_name="Schirrmeister2017", subject_ids=list(range(1, 15)))