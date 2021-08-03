import numpy as np
import matplotlib.pyplot as plt 
import nibabel as nib

sk = nib.load('data/multiskel_vess.nii.gz').get_fdata()
