import multiprocessing as mp
import nibabel as nib
import numpy as np
import os
import sys
import time

from skimage.morphology import skeletonize
from scipy.ndimage.measurements import label
from pathlib import Path

# /Users/giosue/opt/miniconda3/bin/python source/ruler.py data/vessels.nii.gz
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

        self.input_path = Path(path)
    
    def bounding_box(self, img):
        r = np.any(img, axis=(1, 2))
        c = np.any(img, axis=(0, 2))
        z = np.any(img, axis=(0, 1))

        rmin, rmax = np.where(r)[0][[0, -1]]
        cmin, cmax = np.where(c)[0][[0, -1]]
        zmin, zmax = np.where(z)[0][[0, -1]]

        return rmin, rmax, cmin, cmax, zmin, zmax

    def _compute_chunk(self, mask_label_tuple):
        label = mask_label_tuple
        xmin, xmax, ymin, ymax, zmin, zmax = self.bounding_box(self.labeled == label)
        bb = (slice(xmin, xmax+1), slice(ymin,ymax+1), slice(zmin, zmax+1))
        mask = self.input_mask.get_fdata()[bb]
        skel = skeletonize(mask.astype(np.uint8))
        return (skel, label, bb)

    def compute(self, min_voxels = 100):
        print('Importing mask...')
        vessels = self.input_mask.get_fdata()

        print('Dividing mask in connected components...')
        self.labeled, _ = label(vessels, structure=np.ones(shape=(3,3,3)))

        unique, counts = np.unique(self.labeled, return_counts=True)
        labels = unique[counts > min_voxels][1:]

        multilab = np.zeros_like(vessels)
        multiskel = np.zeros_like(vessels)

        print('Creating processes pool...')
        pool = mp.Pool(processes=self.threads)
        chunks = [(l ,) for l in labels]

        print('Starting computation...\n\n')
        skeleton_iterable = pool.imap_unordered(self._compute_chunk, chunks, chunksize=max(1, int(len(labels) / (4*self.threads))))

        n = 1

        for i, res_tuple in enumerate(skeleton_iterable):
            skeleton, l, bb = res_tuple
            print('', end='\r')
            print(f'Completion:  {int(i/len(labels)*100)}%', end='\r')
            if(np.sum(skeleton) > 10):
                multilab = multilab + (self.labeled==l) * n
                multiskel[bb] = skeleton * n
                n += 1
        
        ve = nib.Nifti1Image(multilab.astype(np.float64), self.input_mask.affine)
        sk = nib.Nifti1Image(multiskel.astype(np.float64), self.input_mask.affine)
        os.makedirs(f'tmp/{self.input_path.stem[:-4]}', exist_ok=True)
        nib.save(ve, f'tmp/{self.input_path.stem[:-4]}/multilabel_vessels.nii.gz')
        nib.save(sk, f'tmp/{self.input_path.stem[:-4]}/multilabel_skeleton.nii.gz')

if __name__ == '__main__':
    ruler = Ruler(sys.argv[1])
    tot = time.time()
    ruler.compute()
    print('total: ', time.time() - tot)