import numpy as np
import matplotlib.pyplot as plt 
import nibabel as nib

from scipy.interpolate import splprep, splev

sk = nib.load('data/multiskel_vess.nii.gz').get_fdata()
vess = nib.load('data/multilab_vess.nii.gz').get_fdata()
print(nib.load('data/multiskel_vess.nii.gz').header.get_zooms())
skel = (sk == 2.0).astype(int)
vess = (vess == 2.0).astype(int)

one = np.where(skel==1)

tck, u = splprep([one[0], one[1], one[2]], s=len(one[0])-np.sqrt(2*len(one[0])))
new_points = splev(u, tck)
derivs = splev(u, tck, der=1)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(222, projection='3d')
ax2 = fig.add_subplot(221, projection='3d')
v = np.where(vess == 1)
ax.scatter(one[0], one[1], one[2], c='black', s=1)
ax.scatter(v[0], v[1], v[2], c='red', s=1, alpha=0.5)
ax2.scatter(new_points[0], new_points[1], new_points[2], c='black', s=1)

el = 0

ax.plot([new_points[0][el-1], new_points[0][el + 1]], [new_points[1][el-1], new_points[1][el+1]], [new_points[2][el-1],new_points[2][el+1]])


der = np.array([derivs[0][el], derivs[1][el], derivs[1][el]])
der = der / np.linalg.norm(der)
ax2.plot([new_points[0][el], new_points[0][el] + der[0]*2], [new_points[1][el], new_points[1][el] + der[1]*2], [new_points[2][el],new_points[2][el] + der[2]*2])
ax2.set_title('spliner')
plt.show()

