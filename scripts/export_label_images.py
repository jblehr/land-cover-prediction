from matplotlib import pyplot as plt
import os
from os.path import join as opj
from skimage import io

import utils

DATA_DIR = 'data'
LABELS_DIR = opj(DATA_DIR, 'processed', 'tif', 'labels')


if __name__=='__main__':
    # Saves one of each label mask according to the color coding for Toker et
    # al.  Saves 2018-01-01 by default.  

    aois = os.listdir(LABELS_DIR)
    for aoi in aois:
        print(aoi)
        aoi_path = opj(LABELS_DIR, aoi)
        for date in os.listdir(aoi_path)[:1]: # get one image for now
            mask = io.imread(opj(aoi_path, date))
            rgb_labels = utils.mask2rgb(mask)
            plt.imshow(rgb_labels)
            plt.savefig(opj(DATA_DIR, 'label_images', aoi + '.png'))