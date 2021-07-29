import multiprocessing as mp
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
class Ruler:
    def __init__(self, path, n_threads = None):
        if n_threads is None:
            if(os.cpu_count() > 4):
                self.threads = os.cpu_count() - 1
            else:
                self.threads = os.cpu_count()
            self.input_mask = nib.load(path)
        else:
            self.threads = n_threads
        print(f'Using {self.threads} CPU cores\n')
        self.input_path = Path(path)
        self.test = 0
    
    def _plane(self, grid_shape, point, norm):
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

    def bounding_box(self, img):
        r = np.any(img, axis=(1, 2))
        c = np.any(img, axis=(0, 2))
        z = np.any(img, axis=(0, 1))

        rmin, rmax = np.where(r)[0][[0, -1]]
        cmin, cmax = np.where(c)[0][[0, -1]]
        zmin, zmax = np.where(z)[0][[0, -1]]

        return rmin, rmax, cmin, cmax, zmin, zmax

    def _compute_chunk(self, label):
        # skeletonization
        label, index = label
        xmin, xmax, ymin, ymax, zmin, zmax = self.bounding_box(self.labeled == label)
        bb = (slice(xmin, xmax+1), slice(ymin,ymax+1), slice(zmin, zmax+1))
        mask = self.input_mask.get_fdata()[bb]
        skel = skeletonize(mask.astype(np.uint8))
        areas = np.zeros_like(skel)

        # smoothing + derivatives
        temp_skel = np.where(skel==1)
        if(len(temp_skel[0]) < 11):
            return (skel, areas, label, bb)
        
        tck, u = splprep([temp_skel[0], temp_skel[1], temp_skel[2]], s=len(temp_skel[0])-np.sqrt(2*len(temp_skel[0])))
        derivatives = splev(u, tck, der=1)
        # compute intersection
        PIXDIM = self.input_mask.header.get_zooms()[:3]

        l = []
        for i in range(len(temp_skel[0])):
            cube = skel[max(0,temp_skel[0][i]-1):temp_skel[0][i]+2, max(0, temp_skel[1][i]-1):temp_skel[1][i]+2, max(0, temp_skel[2][i]-1):temp_skel[2][i]+2]
            l.append(np.sum(cube))

        idxs = [i for i, d in enumerate(l) if d == 3]
        for idx in idxs:
            normal = np.array([derivatives[i][idx] for i in range(3)])
            normal = normal / np.linalg.norm(normal)

            o = np.array([temp_skel[0][idx], temp_skel[1][idx], temp_skel[2][idx]])
            p = self._plane(mask.shape, o, normal)
            intersection = p * mask

            labeled, _ = MultiLabel(intersection, structure=np.ones(shape=(3,3,3)))

            lab = labeled[o[0], o[1], o[2]]
            if lab != 0:
                s = PIXDIM[0] * np.sqrt(1 + ((PIXDIM[2]/PIXDIM[0])**2 - 1)*normal[2])
                areas[temp_skel[0][idx], temp_skel[1][idx], temp_skel[2][idx]] = (np.sum(intersection)*PIXDIM[0]*PIXDIM[1]*PIXDIM[2] / s)
        print('', end='\r')
        print(f'Completed idx {index}', end='\r')
        return (skel, areas, label, bb)

    def compute(self, min_voxels = 100):
        print('Importing mask...')
        vessels = self.input_mask.get_fdata()

        print('Dividing mask in connected components...')
        self.labeled, _ = MultiLabel(vessels, structure=np.ones(shape=(3,3,3)))

        unique, counts = np.unique(self.labeled, return_counts=True)
        labels = unique[counts > min_voxels][1:]

        multilab = np.zeros_like(vessels)
        multiskel = np.zeros_like(vessels)
        areas = np.zeros_like(vessels)
        chunks = [(l, i) for i, l in enumerate(labels)]

        print(f'Creating processes pool with {len(chunks)} batches...')
        pool = mp.Pool(processes=self.threads)

        print(f'Starting computation with batch length {max(1, int(len(labels) / (2*self.threads)))}...\n\n')
        skeleton_iterable = pool.map(self._compute_chunk, chunks, chunksize=max(1, int(len(labels) / (2*self.threads))))

        n = 1

        i = 0
        for res_tuple in skeleton_iterable:
            skeleton, area, l, bb = res_tuple
            print('', end='\r\r')
            print(f'Completed batch {i}\nCompletion: {int(i/len(labels)*100)}%', end='\r')
            if(np.sum(skeleton) > 10):
                multilab = multilab + (self.labeled==l) * n
                multiskel[bb] = skeleton * n
                areas[bb] = area
                n += 1
            i += 1
        pool.close()
        pool.join()
        ve = nib.Nifti1Image(multilab.astype(np.float64), self.input_mask.affine)
        sk = nib.Nifti1Image(multiskel.astype(np.float64), self.input_mask.affine)
        ar = nib.Nifti1Image(areas.astype(np.float64), self.input_mask.affine)
        os.makedirs(f'tmp/{self.input_path.stem[:-4]}', exist_ok=True)
        nib.save(ve, f'tmp/{self.input_path.stem[:-4]}/multilabel_vessels.nii.gz')
        nib.save(sk, f'tmp/{self.input_path.stem[:-4]}/multilabel_skeleton.nii.gz')
        nib.save(ar, f'tmp/{self.input_path.stem[:-4]}/areas.nii.gz')

if __name__ == '__main__':
    ruler = Ruler(sys.argv[1])
    tot = time.time()
    ruler.compute()
    print('total: ', time.time() - tot)