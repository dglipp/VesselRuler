from multiprocessing import Process, RawArray, Value
import nibabel as nib
import numpy as np
import os
import sys
import time

from skimage.morphology import skeletonize
from scipy.ndimage.measurements import label as MultiLabel
from scipy.interpolate import splprep, splev
from pathlib import Path

# /Users/giosue/opt/miniconda3/bin/python source/ruler.py data/vessels.nii.gz

# python -m nuitka --standalone  --plugin-enable=numpy --plugin-enable=pylint-warnings --plugin-enable=pkg-resources ruler.py

def _plane(grid_shape, point, norm):
        point = np.array(point)
        norm = np.array(norm)
        d = -point.dot(norm)

        plane_labelmap = np.zeros(grid_shape)

        if norm[1:].tolist() == [0, 0]:
            plane_labelmap[point[0], :, :] = 1
            return plane_labelmap

        if norm[2] == 0:
            xx = list(range(0, grid_shape[0]))
            yy = np.array(point[1] - norm[0] / norm[1] * (xx - point[0]), dtype=int)
            for i in range(len(xx)):
                if yy[i] in range(grid_shape[1]):
                    plane_labelmap[xx[i], yy[i], :] = 1
            return plane_labelmap

        xx, yy = np.meshgrid(range(grid_shape[0]), range(grid_shape[1]), indexing='ij')


        z = ((-norm[0] * xx - norm[1] * yy - d) /norm[2]).astype(int)

        arr = np.array([[_x, _y, z[_x, _y]] for _x in range(grid_shape[0]) for _y in range(grid_shape[1])])


        for r in arr:
            if r[2] in range(grid_shape[2]):
                plane_labelmap[r[0], r[1], r[2]] = 1
        plane_labelmap = np.round(plane_labelmap).astype(int)
    
        return plane_labelmap

def _bounding_box(img):
    r = np.any(img, axis=(1, 2))
    c = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))

    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    return rmin, rmax, cmin, cmax, zmin, zmax

def _create_chunks(lst, n_chunks):
    chunk_length = len(lst) // n_chunks
    new = [[lst[i] for i in range(l*chunk_length, (l+1)*chunk_length)] for l in range(n_chunks)]
    if len(lst) % n_chunks:
        last = [(lst[-i]) for i in range(1, len(lst) % n_chunks + 1)]
        last.reverse()
        new.append(last)
    return new

def _compute_chunk(input_array, output_array, label_array, pixdim, shape):
    # skeletonization
    labeled = np.frombuffer(input_array, dtype=np.float64).reshape(shape)
    areas_buffer = np.frombuffer(output_array, dtype=np.float64).reshape(shape)
    for label in label_array:
        mask = labeled == label
        xmin, xmax, ymin, ymax, zmin, zmax = _bounding_box(mask)
        bb = (slice(xmin, xmax+1), slice(ymin,ymax+1), slice(zmin, zmax+1))
        mask = mask[bb]
        skel = skeletonize(mask.astype(np.uint8))
        areas = np.zeros_like(skel)

        # smoothing + derivatives
        temp_skel = np.where(skel==1)
        if(len(temp_skel[0]) < 11):
            return
        
        tck, u = splprep([temp_skel[0], temp_skel[1], temp_skel[2]], s=len(temp_skel[0])-np.sqrt(2*len(temp_skel[0])))
        derivatives = splev(u, tck, der=1)
        # compute intersection

        l = []
        for i in range(len(temp_skel[0])):
            cube = skel[max(0,temp_skel[0][i]-1):temp_skel[0][i]+2, max(0, temp_skel[1][i]-1):temp_skel[1][i]+2, max(0, temp_skel[2][i]-1):temp_skel[2][i]+2]
            l.append(np.sum(cube))

        idxs = [i for i, d in enumerate(l) if d == 3]
        for idx in idxs:
            normal = np.array([derivatives[i][idx] for i in range(3)])
            normal = normal / np.linalg.norm(normal)

            o = np.array([temp_skel[0][idx], temp_skel[1][idx], temp_skel[2][idx]])
            p = _plane(mask.shape, o, normal)
            intersection = p * mask

            labeled, _ = MultiLabel(intersection, structure=np.ones(shape=(3,3,3)))

            lab = labeled[o[0], o[1], o[2]]
            if lab != 0:
                s = pixdim[0] * np.sqrt(1 + ((pixdim[2]/pixdim[0])**2 - 1)*normal[2])
                areas[temp_skel[0][idx], temp_skel[1][idx], temp_skel[2][idx]] = (np.sum(intersection)*pixdim[0]*pixdim[1]*pixdim[2] / s)
        areas_buffer[bb] = areas

def main(path, min_voxels = 100):
    if(os.cpu_count() > 4):
        threads = os.cpu_count() - 1
    else:
        threads = os.cpu_count()
    print(f'Using {threads} CPU cores\n')

    
    print('Importing mask...')
    path = Path(path)
    input_mask = nib.load(path)

    vessels = input_mask.get_fdata()

    print('Dividing mask in connected components...')
    multilabeled = MultiLabel(vessels, structure=np.ones(shape=(3,3,3)))[0].astype(int)

    os.makedirs(f'tmp/{path.stem[:-4]}', exist_ok=True)
    ve = nib.Nifti1Image(multilabeled.astype(np.float64), input_mask.affine)
    nib.save(ve, f'tmp/{path.stem[:-4]}/multilabel_vessels.nii.gz')

    print('Removing too small vessels...')
    unique, counts = np.unique(multilabeled, return_counts=True)
    labels = unique[counts > min_voxels][1:]
    for l in np.setdiff1d(unique, labels):
        multilabeled[multilabeled == l] = 0
    
    print('Allocating shared memory...')
    length = multilabeled.reshape(-1,1).shape[0]
    shared_mask = RawArray('d', multilabeled.reshape(-1,1))
    shared_skel = RawArray('d', length)
    chunks = _create_chunks(labels, max(1, int(len(labels) / (2*threads))))

    print(f'Creating processes pool with {len(chunks)} batches...')
    pool = [Process(target=_compute_chunk, args=(shared_mask, shared_skel, c, input_mask.header.get_zooms()[:3], multilabeled.shape)) for c in chunks]

    print(f'Starting computation with batch length {max(1, int(len(labels) / (2*threads)))}...\n\n')
    for p in pool:
        p.start()

    for p in pool:
        p.join()

    ar = nib.Nifti1Image(np.frombuffer(shared_skel, dtype=np.float64).reshape(vessels.shape), input_mask.affine)
    nib.save(ar, f'tmp/{path.stem[:-4]}/areas.nii.gz')
    #sk = nib.Nifti1Image(multiskel.astype(np.float64), self.input_mask.affine)
    #nib.save(sk, f'tmp/{self.input_path.stem[:-4]}/multilabel_skeleton.nii.gz')

if __name__ == '__main__':
    tot = time.time()
    main(sys.argv[1])
    print('total: ', time.time() - tot)