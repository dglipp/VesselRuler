import numpy as np
import time

zipped = [(1, 1, 1), (2, 2, 2), (33, 44, 66)]
unzipped_object = zip(*zipped)
unzipped_list = list(unzipped_object)

grid_shape = (512, 512, 512)
labelmap1 = np.zeros(grid_shape)
norm = [1,2,3]
d = 4
xx, yy = np.meshgrid(range(grid_shape[0]), range(grid_shape[1]), indexing='ij')

z = -((-norm[0] * xx - norm[1] * yy - d) /norm[2]).astype(int)



print('starting')
ts = time.time()
arr = np.array([[_x, _y, z[_x, _y]] for _x in range(grid_shape[0]) for _y in range(grid_shape[1])])

for r in arr:
    if r[2] in range(grid_shape[2]):
        labelmap1[r[0], r[1], r[2]] = 1
print('original', time.time()-ts)

labelmap2 = np.zeros(grid_shape)
ts = time.time()

arr = ([], [], [])
for _x in range(grid_shape[0]):
    for _y in range(grid_shape[1]):
        zed = z[_x, _y]
        if zed in range(grid_shape[2]):
            arr[0].append(_x)
            arr[1].append(_y)
            arr[2].append(zed)

labelmap2[arr] = 1

print('new', time.time()-ts)

ts = time.time()
labelmap3 = np.zeros(grid_shape)


print('defin', time.time()- ts)