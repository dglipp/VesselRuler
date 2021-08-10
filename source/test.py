import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

path = 'data/3'
vess = nib.load(f'results/{path}/BV_classes.nii.gz').get_fdata()
areas = nib.load(f'tmp/{path}/areas.nii.gz').get_fdata()

def col(v):
    if v == 2:
        return 'red'
    if v == 3:
        return 'green'
    if v == 4:
        return 'yellow'

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

pos = np.where(vess!=0)
ax.scatter(pos[0], pos[1], pos[2], c=list(map(col, vess[pos])), alpha=0.6, s = 0.1)

ax = fig.add_subplot(333)
val = np.delete(areas.flatten(), areas.flatten() <= 1.5)
print(np.sum(vess == 2) / np.sum(vess >= 2), np.sum(vess == 3) / np.sum(vess >= 2), np.sum(vess == 4) / np.sum(vess >= 2))
sns.histplot(val, kde=True, binwidth=1, ax=ax)
ax.set_xlim(0, 100)

plt.show()
