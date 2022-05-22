from itertools import product
from pyexpat.model import XML_CQUANT_PLUS
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import numpy as np
import os
import datetime
from torchvision import transforms
import cv2

class FullyIndependentDataset(Dataset):
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

def normalize(cube):
    # Normalize
    norm = np.zeros((cube.shape[0],cube.shape[1]))
    final_cube = cv2.normalize(cube,  norm, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return final_cube

class SpatiotemporalDataset(Dataset):
    def __init__(
        self,
        data_dir,
        poi_list,
        n_steps,
        dims,
        cell_width=16,
        transform=None,
        labs_as_features=False,
        normalize=False
    ):
        self.poi_list = poi_list
        self.cell_width = cell_width
        self.data_dir = data_dir
        self.n_steps = n_steps
        self.transform = transform
        self.labs_as_features = labs_as_features
        self.n_ambigious = 0
        self.x_dim, self.y_dim = dims

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
            
            if normalize:
                rgb_cubes = [normalize(cube) for cube in rgb_cubes]

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

        assert rgb_st.shape[2] % self.cell_width == 0
        assert rgb_st.shape[3] % self.cell_width == 0

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
        # In new method, we split the 1024x1024 image into individual cells of 
        # width cell_width. So, we have 1024/cell_width * 1024/cell_width obs
        # return int(self.x_dim / self.cell_width) * \
            # int(self.y_dim / self.cell_width) *
        return len(self.poi_list)

    def __getitem__(self, idx):

        cube_path = self.poi_list[idx]
        datX, datY = self.read_hypercube(poi_name=cube_path)
        x_steps = np.arange(0, datX.shape[2], self.cell_width)
        y_steps = np.arange(0, datX.shape[3], self.cell_width)

        locs = product(x_steps, y_steps)

        out_tensor_X = np.empty(shape = (
            int(datX.shape[2]/self.cell_width * datX.shape[3]/self.cell_width), #Batch size (sliced image)
            self.n_steps, # Number of timesteps
            datX.shape[1], # N channels
            self.cell_width, 
            self.cell_width
        ))

        out_tensor_Y = np.empty(shape = (
            int(datX.shape[2]/self.cell_width * datX.shape[3]/self.cell_width), #Batch size (sliced image)
            self.n_steps, # Number of timesteps
            self.cell_width, 
            self.cell_width
        ))

        for batch_idx, loc_idx in enumerate(locs):

            x_pix_min, y_pix_min = loc_idx
            x_pix_max = x_pix_min + self.cell_width
            y_pix_max = y_pix_min + self.cell_width

            x_core = datX[:, :, x_pix_min:x_pix_max, y_pix_min:y_pix_max]
            y_core = datY[:, x_pix_min:x_pix_max, y_pix_min:y_pix_max]

            out_tensor_X[batch_idx, :, :, :, : ] = x_core
            out_tensor_Y[batch_idx, :, :, : ] = y_core

        return (
            torch.tensor(out_tensor_X.astype(np.float32)),
            torch.tensor(out_tensor_Y).long()
        )


if __name__ == "__main__":
    poi_list = ['1700_3100_13_13N', '2065_3647_13_16N', '2697_3715_13_20N']
    STData = SpatiotemporalDataset(
        "data/processed/npz",
        poi_list=poi_list,
        cell_width=128,
        dims = (1024,1024),
        n_steps=5
    )
    outX, outY = STData.__getitem__(2)