import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

vss = nib.load('tmp/data/175252_20200314/areas.nii.gz')
affine = vss.affine
vess = vss.get_fdata()

def col(v):
    if v <= 5:
        return 'red'
    if v <= 10:
        return 'green'
    if v > 10:
        return 'yellow'

pixdim = vss.header.get_zooms()[:3]
s = pixdim[0] * np.sqrt(1 + ((pixdim[2]/pixdim[0])**2 - 1)*0.7)
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

pos = np.where(vess!=0)
ax.scatter(pos[0], pos[1], pos[2], c=list(map(col, vess[pos])), alpha=0.6, s = 0.1)

ax = fig.add_subplot(333)
sns.histplot(np.delete(vess.flatten(), vess.flatten() <= 1.5), kde=True, bins=100, ax=ax)
ax.set_xlim(0, 140)

plt.show()
