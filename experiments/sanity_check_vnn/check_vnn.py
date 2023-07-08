import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import os
import glob
import nibabel as nib
from datetime import datetime
import random
import pdb

import medutils

import torch

from pytorch3d.transforms import RotateAxisAngle, Rotate, random_rotations

from models_mnist import get_model, get_model_complex


classifier = get_model()
classifier.load_state_dict(torch.load('/scratch/users/shizhehe/model_pc.pt'))


from pytorch3d.transforms import RotateAxisAngle, Rotate, random_rotations

x = torch.rand(64, 64, 3)
scores, feature_space, feat = classifier(x)

trot = RotateAxisAngle(angle=90, axis="Z", degrees=True)
fs_tr = trot.transform_points(feature_space)

print(fs_tr.shape)

x2 = x.detach().clone()
x2 = trot.transform_points(x2)
print(x2.shape)

scores2, feature_space2, feat2 = classifier(x2)

print(feature_space2 - fs_tr)