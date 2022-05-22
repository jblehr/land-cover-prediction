import rasterio
import numpy as np
import os
import shutil
from matplotlib import pyplot as plt
from rasterio.plot import show
import glob
from os.path import join as opj
import sys

# ### Data organization
# Goal is to resave files from their downloaded form into the following organization
# to ensure each label file can be matched with the planet data

# data  
#     / processed
#         / tif  
#             / labels  
#                 / aoi (area of interest id)  
#                     date.tif  
#            / planet  
#                / aoi  
#                     date.tif  
#         / npz  
#             / labels  
#                 / aoi  
#                     date.npz  
#             / planet  
#                 / aoi  
#                     date.npz 


DATA_DIR = 'data'
RAW_DIR = opj(DATA_DIR, 'raw')
PROCESSED_DIR = opj(DATA_DIR, 'processed')

# Helper functions for resaving files
def copy_over_file(full_path, proc_dir, fldr):
    new_name = full_path[-14:]
    fldr_path = opj(proc_dir, fldr)
    make_new_path(fldr_path)
    shutil.copyfile(full_path, opj(fldr_path, new_name))

def make_new_path(new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

def process_labels():
    raw_path = opj(RAW_DIR, 'labels', 'labels')
    if os.path.exists(raw_path):
        label_files = list({f for f in glob.glob(raw_path + "**/**/**.tif", recursive=True)})

        proc_labels_dir = opj(PROCESSED_DIR, 'tif', 'labels')
        make_new_path(proc_labels_dir)

        for full_path in label_files:
            # Extract folder name
            fldr = full_path.replace(raw_path,"").split('/')[0]
            copy_over_file(full_path, proc_labels_dir, fldr)

        #We should have (12 months * 2 years * 55 locations) .tif files
        proc_files = {f for f in glob.glob(proc_labels_dir + "**/**/**.tif", recursive=True)}
        n_files = len(proc_files)
        print("n files", n_files)
        assert n_files == 55 * 24

        # Remove labels and labels.zip
        shutil.rmtree(opj(RAW_DIR, 'labels'))
        os.remove(opj(RAW_DIR, 'labels.zip'))

def get_list_of_aois_without_labels():
    '''
    There are 20 AOIS without corresponding labels. We do not use these.
    '''
    files = []
    for name in ['test.txt', 'val.txt']:
        with open(os.path.join(DATA_DIR, name), 'r') as file:
            files.extend(file.readlines())
    return [f.strip() for f in files]


def process_planet(planet_file, no_labels_files):
    '''
    planet_file: planet.22S # name after unzipped
    '''
    raw_path = opj(RAW_DIR, planet_file)
    if not os.path.exists(raw_path):
        return []

    planet_files = list({f for f in glob.glob(f'{RAW_DIR}/{planet_file}/**/**/**.tif', recursive=True)})
    proc_data_dir = opj(PROCESSED_DIR, 'tif', 'planet')
    make_new_path(proc_data_dir)
    fldr_list = set()
    for full_path in planet_files:
        # remove last 21 chars
        aoi = full_path[-33:-21]
        # Do not include files that do not have corresponding labels
        if aoi not in no_labels_files:
            digits = full_path.replace(RAW_DIR + '/planet.', "")[:3]
            fldr = aoi+'_' + digits
            copy_over_file(full_path, proc_data_dir, fldr)
            fldr_list.add(fldr)


    # We should have 2 * 12 .tif files for each AOI
    proc_files = {f for f in glob.glob(proc_data_dir + "**/**/**.tif", recursive=True)}
    n_files = len(proc_files)
    assert n_files == len(os.listdir(proc_data_dir)) * 24

    # Remove remaining raw data
    shutil.rmtree(opj(RAW_DIR, planet_file))

    # return list of folders
    return list(fldr_list)


def save_as_npz(fldr_list):
    '''
    Save both labels and planet as npz files
    '''
    for type in ['labels', 'planet']:
        # Only get data for just-processed files
        for aoi in fldr_list:
            new_dir = opj(PROCESSED_DIR, 'npz', type, aoi)
            make_new_path(new_dir)
            # Only get monthly file name from label dir
            for tif_file in os.listdir(opj(PROCESSED_DIR, 'tif', 'labels', aoi)):
                tif_path = opj(PROCESSED_DIR, 'tif', type, aoi, tif_file)
                # read as np ndarrays
                try:
                    tif_as_np = rasterio.open(tif_path).read()
                except:
                    tif_path = opj(PROCESSED_DIR, 'tif', type, aoi, tif_file.replace('_','-'))
                    tif_as_np = rasterio.open(tif_path).read()

                npz_path = opj(new_dir, tif_file.replace('_','-')).replace('.tif', '')
                np.savez_compressed(npz_path, tif_as_np)

if __name__=="__main__":
    planet_file = sys.argv[1]
    print('Processing labels')
    process_labels()
    print('Processing planet')
    no_labels_files = get_list_of_aois_without_labels()
    fldr_list = process_planet(planet_file, no_labels_files)
    print(fldr_list)
    print('Saving as npz')
    save_as_npz(fldr_list)
