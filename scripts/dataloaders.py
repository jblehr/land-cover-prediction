from itertools import product
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import numpy as np
import os
import datetime
from torchvision import transforms
# import cv2
# import s3_util

class FullyIndependentDataset(Dataset):
    """
    Custom torch Dataset class that considers each pixel independently. Originally
    used in first iteration of Logistic Regression, but now depreciated (LReg
    uses same workflow as the convGRU).
    """
    def __init__(self, img_path, label_path):

        # Grouping variable names
        self.categorical = ["0", "1", "2", "3", "4", "5", "6"]
        self.target = "land_class"
        self.n_ambigious = 0

        img_cube = np.load(img_path)["arr_0"]
        label_cube = np.load(label_path)["arr_0"]

        fi_rgbn = np.empty(shape=(img_cube.shape[1] * img_cube.shape[2], 4))
        fi_class = np.empty(shape=(img_cube.shape[1] * img_cube.shape[2],))

        fi_idx = 0
        for x_pos in range(img_cube.shape[1]):
            for y_pos in range(img_cube.shape[2]):
                fi_rgbn[fi_idx, :] = img_cube[:, x_pos, y_pos]
                one_hot = (label_cube[:, x_pos, y_pos] == 255).astype(int)
                if sum(one_hot) > 1:
                    one_ind = np.where(one_hot == 1)[0]
                    one_hot[one_ind[1:]] = 0
                    self.n_ambigious += 1
                # fi_class[fi_idx, :] = one_hot
                fi_class[fi_idx] = torch.tensor(np.where(one_hot == 1)[0])
                fi_idx += 1

        # Save target and predictors
        self.X = torch.tensor(fi_rgbn.astype(np.float32))
        self.y = torch.tensor(fi_class).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (self.X[idx, :], self.y[idx])

# def normalize(cube):
#     # Normalize
#     norm = np.zeros((cube.shape[0],cube.shape[1]))
#     final_cube = cv2.normalize(cube,  norm, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
#     return final_cube

class SpatiotemporalDataset(Dataset):
    def __init__(
        self,
        data_dir,
        poi_list,
        n_steps,
        dims,
        cell_width_pct,
        transform=None,
        labs_as_features=False,
        download=False,
        in_memory=False
    ):
        """A dataloader that maintains the Spatiotemporal structure of the data.
        Organized as (Batch, Time, Channels, XDim, YDim).

        Args:
            data_dir (str): dir to load data from
            poi_list (list of str): which POIs to load
            n_steps (int): number of steps to load. When this is 2, there is no
                time component to modelling (since each batch predicts the next Y)
            dims (dims): original dims of the input data (should be 1024)
            cell_width_pct (float): percent of input image to split into individ
                obs. For eg if input dim is 1024 and cell_width_pct is .5, split
                will be 4 obs of 512x512. If == 1, no splitting. 
            transform (torchvision transforms, optional): torchvision transforms
                to apply.
            labs_as_features (bool, optional): if true, instead of passing input
                channels, pass previous class labels as features. Makes classifying
                trivial as most don't change, so good proof of concept.
            download (bool, optional): if True, download data from S3. Doesn't 
                work on UChicago servers.
            in_memory (bool, optional): if True, load all data in memory during
                training. If the node doesn't blow up, greatly reduces time spent
                between train steps. 
        """

        if download:
            print('Downloading data from S3...')
            s3_util.download_npz_dir(
                'capp-advml-land-use-prediction-small',
                poi_list,
                'data/processed'
            )
            print('Download complete!')
        self.hypercubes = {}
        self.poi_list = poi_list
        self.cell_width_pct = cell_width_pct
        self.data_dir = data_dir
        self.n_steps = n_steps
        self.transform = transform
        self.labs_as_features = labs_as_features
        self.n_ambigious = 0
        self.x_dim, self.y_dim = dims
        self.in_memory = in_memory

        if in_memory:
            for poi in poi_list:
                datX, datY = self.read_hypercube(poi)
                self.hypercubes[poi] = (datX, datY)

    def read_hypercube(self, poi_name):
        rgb_dir = os.path.join(self.data_dir, "planet", poi_name)
        rgb_files = sorted(os.listdir(rgb_dir))

        rgb_files = [os.path.join(rgb_dir, rgb_file) for rgb_file in rgb_files]
        rgb_files = rgb_files[:self.n_steps]

        lab_dir = os.path.join(self.data_dir, "labels", poi_name)
        lab_files = sorted(os.listdir(lab_dir))
        lab_files = [os.path.join(lab_dir, lab_file) for lab_file in lab_files]
        lab_files = lab_files[:self.n_steps]

        # ensure no mismatch between cubes
        assert all(
            [
                rgb_files[idx] == lab_files[idx].replace("labels", "planet")
                for idx in range(len(rgb_files))
            ]
        )

        lab_cubes = [np.load(lab_file)["arr_0"] for lab_file in lab_files]

        if not self.labs_as_features:
            rgb_cubes = [np.load(rgb_file)["arr_0"] for rgb_file in rgb_files]

            # rgb needs no preprocessing, so just stack
            rgb_st = np.stack(rgb_cubes)
        else:
            # for our sanity check, we leave the labels as 7 channels, which should
            # give the model a massive leg up on predicting the next step since 
            # there are very few changes from month to month. Ie, just use AR
            # order 1 for each dummy, and that will give you the next value
            rgb_st = np.stack(lab_cubes)

        # need to get idx label in one mask from labels, then stack
        # n_ambigious increases when we have to arbitrarily choose a class
        # because there are multiple nonzeros in the land class masks
        lab_cubes = [self.collapse_labels(lab_cube) for lab_cube in lab_cubes]
        lab_st = np.stack(lab_cubes)

        # save target and predictors
        X = torch.tensor(rgb_st.astype(np.float32))
        if self.transform:
            X = self.transform(X)
        Y = torch.tensor(lab_st).squeeze().long()

        return (X, Y)

    def collapse_labels(self, label_7d):
        '''
        Convert 7 dim label mask to (1, m, m) by assigning index value
        '''
        # dim = 1 for final labels
        label_1d = np.empty((1, label_7d.shape[1], label_7d.shape[2])) 
        for i in range(label_7d.shape[1]):
            for j in range(label_7d.shape[2]):
                for k in range(label_7d.shape[0]):
                    # TODO: consider another way to break ties. Right now, only select first value == 255
                    if label_7d[k, i, j] == 255:
                        label_1d[0, i, j] = k
                        break 
        return label_1d.astype(np.uint8)

    def __len__(self):
        return len(self.poi_list)

    def __getitem__(self, idx):

        poi = self.poi_list[idx]
        if not self.in_memory:
            datX, datY = self.read_hypercube(poi_name=poi)
        else:
            datX, datY = self.hypercubes[poi]

        transformed_cell_width = int(datX.shape[2] * self.cell_width_pct)
        original_cell_width = int(self.x_dim * self.cell_width_pct)

        assert transformed_cell_width == datX.shape[3] * self.cell_width_pct
        assert original_cell_width == self.y_dim * self.cell_width_pct

        x_steps = np.arange(0, datX.shape[2], transformed_cell_width)
        y_steps = np.arange(0, datX.shape[3], transformed_cell_width)


        locs = product(x_steps, y_steps)
        batch_size = \
            int(datX.shape[2]/transformed_cell_width * 
                datX.shape[3]/transformed_cell_width)

        # Here and below, we want to pass the transformed x images but the 
        # full target images. The model itself will upsample back to the 
        # proper dimensions in the output

        out_tensor_X = np.empty(shape = (
            batch_size, #Batch size (sliced image)
            self.n_steps, # Number of timesteps
            datX.shape[1], # N channels
            transformed_cell_width, 
            transformed_cell_width
        ))

        out_tensor_Y = np.empty(shape = (
            batch_size, #Batch size (sliced image)
            self.n_steps, # Number of timesteps
            original_cell_width, 
            original_cell_width
        ))

        for batch_idx, loc_idx in enumerate(locs):

            x_pix_min, y_pix_min = loc_idx

            x_pix_max = x_pix_min + transformed_cell_width
            y_pix_max = y_pix_min + transformed_cell_width
            x_core = datX[:, :, x_pix_min:x_pix_max, y_pix_min:y_pix_max]

            x_pix_max = x_pix_min + original_cell_width
            y_pix_max = y_pix_min + original_cell_width
            y_core = datY[:, x_pix_min:x_pix_max, y_pix_min:y_pix_max]

            out_tensor_X[batch_idx, :, :, :, : ] = x_core
            out_tensor_Y[batch_idx, :, :, : ] = y_core

        return (
            torch.tensor(out_tensor_X.astype(np.float32)),
            torch.tensor(out_tensor_Y).long()
        )

if __name__ == '__main__':
    STData = SpatiotemporalDataset(
        "data/processed/npz",
        dims = (1024, 1024), #Original dims, not post-transformation
        poi_list=['2697_3715_13_20N', '5989_3554_13_44N'],
        n_steps=2, # start with one year
        cell_width_pct=.5,
        labs_as_features=False,
        transform=None,
        download=True,
        in_memory=True
    )
    STData.__getitem__(0)