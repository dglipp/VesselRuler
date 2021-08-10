import numpy as np
import matplotlib.pyplot as plt 
import nibabel as nib
from numpy.core.fromnumeric import shape

def majority(window):
    window = np.array(window)
    window[window == 1] = 0
    unique, count = np.unique(window, return_counts=True)
    arr = np.vstack([unique, count])
    arr = np.delete(arr, arr[0] == 0, axis=1)
    try:
        r = arr[0][np.argmax(np.delete(arr, arr[0] == 0, axis=1)[1])]
    except ValueError:
        r = 1
    return r

areas = nib.load('tmp/data/101079_20200322/areas.nii.gz').get_fdata()
skeleton = nib.load('tmp/data/101079_20200322/multilabel_skeleton.nii.gz')

sk = skeleton.get_fdata()

areas[(areas > 0) * (areas <= 5)] = 2
areas[(areas <= 10) * (areas > 5)] = 3
areas[areas > 10] = 4

sk[sk != 0] = 1
res = sk*(areas == 0) + areas

x, y, z = np.where(res == 1)
v = np.zeros_like(x)

prev = len(x)
removed = 10
while(removed > 0):
    for i in range(len(x)):
        bb = (slice(x[i]-1, x[i]+2), slice(y[i]-1, y[i]+2), slice(z[i]-1, z[i]+2))
        v[i] = majority(res[bb])

    res[x, y, z] = v

    x, y, z = np.where(res == 1)
    v = np.zeros_like(x)
    removed = prev - len(x)
    print(f'removed {removed} elements')
    prev = len(x)

res[res == 1] = 0

