from multiprocessing import Process, Queue, RawValue
import nibabel as nib
import numpy as np
import os
import sys
import time

from skimage.morphology import skeletonize
from scipy.ndimage.measurements import label as MultiLabel
from scipy.interpolate import splprep, splev
from pathlib import Path

# /Users/giosue/opt/miniconda3/bin/python source/ruler_queue.py data/vessels.nii.gz

# python -m nuitka --standalone  --plugin-enable=numpy --plugin-enable=pylint-warnings --plugin-enable=pkg-resources ruler.py

def _plane(grid_shape, point, norm):
        point = np.array(point)
        norm = np.array(norm)
        d = -point.dot(norm)

        plane_labelmap = np.zeros(grid_shape, dtype=int)

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

        z = ((-norm[0] * xx - norm[1] * yy - d) /norm[2]).astype(int).flatten()

        xx = xx.flatten()
        yy = yy.flatten()
        bitm = np.isin(z, range(grid_shape[2]), invert=True)
        plane_labelmap[np.delete(xx, bitm), np.delete(yy, bitm), np.delete(z, bitm)] = 1
    
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

def _compute_chunk(input_queue, output_queue, pixdim, cntinue):
    # skeletonization
    while cntinue.value:
        id, mask = input_queue.get()
        if id == -1:
            return

        if mask is not None:
            skel = skeletonize(mask.astype(np.uint8))
            areas = np.zeros_like(skel)

            # smoothing + derivatives
            temp_skel = np.where(skel==1)
            if(len(temp_skel[0]) < 11):
                output_queue.put((id, None))
                continue
            
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
            output_queue.put((id, areas))
        output_queue.put((id, None))
        
def main(path, min_voxels = 100):
    ts = time.time()
    if(os.cpu_count() > 4):
        threads = os.cpu_count() - 1
    else:
        threads = os.cpu_count()
    print(f'Using {threads} CPU cores\n')

    print('Importing mask...')
    path = Path(path)
    input_mask = nib.load(path)

    vessels = input_mask.get_fdata()
    areas = np.zeros_like(vessels)

    print('Dividing mask in connected components...')
    multilabeled = MultiLabel(vessels, structure=np.ones(shape=(3,3,3)))[0].astype(int)

    print(f'Removing vessels with n_voxels < {min_voxels}...')
    unique, counts = np.unique(multilabeled, return_counts=True)
    sort_idx = np.argsort(counts[1:])
    unique = unique[1:][sort_idx]
    counts = counts[1:][sort_idx]
    labels = list(unique[counts > min_voxels])
    multilabeled *= np.isin(multilabeled, labels)

    print(f'Creating processes pool...')
    input_queue = Queue()
    output_queue = Queue()
    is_end = RawValue('i')
    is_end.value = 1
    pool = [Process(target=_compute_chunk, args=(input_queue, output_queue, input_mask.header.get_zooms()[:3], is_end)) for _ in range(threads)]

    for p in pool:
        p.daemon = True
        p.start()

    print('Starting computation...')
    cnt = 0
    chunk_dict = {}

    tot = len(labels)
    n_done = 0
    doing_ids = []

    while len(labels) > 0:
        if cnt < threads + 1:
            l = labels.pop()
            mask = multilabeled == l
            xmin, xmax, ymin, ymax, zmin, zmax = _bounding_box(mask)
            bb = (slice(xmin, xmax+1), slice(ymin,ymax+1), slice(zmin, zmax+1))
            chunk_dict[l] = bb
            input_queue.put((l, mask[bb]))
            doing_ids.append(l)
            print('', end='\r')
            print(f'Progress: {int(n_done/tot*100)}%', doing_ids, end='\r')
            cnt += 1
        try:
            id, output = output_queue.get(block=True, timeout=3)
            if id != -1:
                try:
                    doing_ids.remove(id)
                except:
                    print(id)
            if output is None:
                multilabeled[multilabeled == id] = 0
            else:
                areas[chunk_dict[id]] = output

            n_done += 1
            cnt -= 1
        except:
            pass

    
    while cnt != 0:
        id, output = output_queue.get()
        if id != -1:
            doing_ids.remove(id)
        if output is None:
            multilabeled[multilabeled == id] = 0
        else:
            areas[chunk_dict[id]] = output
        print('', end='\r')
        print(f'Progress: {int(n_done/tot*100)}%', doing_ids, end='\r')
        n_done += 1
        cnt -= 1

    is_end.value = 0
    for p in pool:
        p.join()

    os.makedirs(f'tmp/{path.stem[:-4]}', exist_ok=True)

    print('\n\nSaving results in tmp folder...')
    ar = nib.Nifti1Image(areas.astype(np.float64), input_mask.affine)
    nib.save(ar, f'tmp/{path.stem[:-4]}/areas.nii.gz')

    sk = nib.Nifti1Image((areas.astype(bool) * multilabeled).astype(np.float64), input_mask.affine)
    nib.save(sk, f'tmp/{path.stem[:-4]}/multilabel_skeleton.nii.gz')

    ve = nib.Nifti1Image(multilabeled.astype(np.float64), input_mask.affine)
    nib.save(ve, f'tmp/{path.stem[:-4]}/multilabel_vessels.nii.gz')

    print(f'Task completed in {round((time.time()-ts)/60)}...')

if __name__ == '__main__':
    main(sys.argv[1])