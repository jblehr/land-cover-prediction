from torch.utils.data import Dataset, DataLoader, random_split
import torch
import numpy as np
import os
import datetime
from torchvision import transforms

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


class SpatiotemporalDataset(Dataset):
    def __init__(
        self,
        data_dir,
        poi_name,
        n_steps,
        bandwidth=10,
        transform=None
    ):
        self.bandwidth = bandwidth
        rgb_dir = os.path.join(data_dir, "planet", poi_name)
        rgb_files = os.listdir(rgb_dir)

        date_range = [
            datetime.datetime.strptime(rgb_file.replace(".npz", ""), "%Y-%m-%d")
            for rgb_file in rgb_files
        ]

        rgb_files = [os.path.join(rgb_dir, rgb_file) for rgb_file in rgb_files]
        rgb_files = rgb_files[:n_steps]

        lab_dir = os.path.join(data_dir, "labels", poi_name)
        lab_files = os.listdir(lab_dir)
        lab_files = [os.path.join(lab_dir, lab_file) for lab_file in lab_files]
        lab_files = lab_files[:n_steps]

        self.date_range = date_range[:n_steps]

        # ensure no mismatch between cubes
        assert all(
            [
                rgb_files[idx] == lab_files[idx].replace("labels", "planet")
                for idx in range(len(rgb_files))
            ]
        )

        rgb_cubes = [np.load(rgb_file)["arr_0"] for rgb_file in rgb_files]
        lab_cubes = [np.load(lab_file)["arr_0"] for lab_file in lab_files]

        # rgb needs no preprocessing, so just stack
        rgb_st = np.stack(rgb_cubes)

        # need to get idx label in one mask from labels, then stack
        # n_ambigious increases when we have to arbitrarily choose a class
        # because there are multiple nonzeros in the land class masks
        self.n_ambigious = 0
        lab_cubes = [
            np.apply_along_axis(self.idx_from_multimask, 0, lab_cube)
            for lab_cube in lab_cubes
        ]
        lab_st = np.stack(lab_cubes)

        # save target and predictors
        self.X = torch.tensor(rgb_st.astype(np.float32))
        self.original_shape = rgb_st.shape
        self.transform = transform
        if self.transform:
            self.X = self.transform(self.X)
        self.y = torch.tensor(lab_st).squeeze().long()
        pass

    def idx_from_multimask(self, arr_1d):
        # TODO: 1.7% ambiguity in the first tried... need a better solution
        land_class = np.where(arr_1d == 255)[0]
        if len(land_class) > 1:
            land_class = land_class[0]
            self.n_ambigious += 1
        return land_class

    def __len__(self):
        # each pixel is an observation, but we will also pass its past and neighbors
        # also remove padding since we don't want to directly sample those edge pixels,
        # it is merely to prevent the edges from causing issues
        return self.original_shape[2] * self.original_shape[3]

    def __getitem__(self, idx):
        # use index to get original x,y coords, then slice the padded pixel
        x_loc = idx // self.original_shape[2]
        x_min, x_max = x_loc, x_loc + self.bandwidth*2 + 1

        y_loc = idx % self.original_shape[3]
        y_min, y_max = y_loc, y_loc + self.bandwidth*2 + 1

        # get the center pixel and all it's bandwidth neighbors
        x_core = self.X[:, :, x_min:x_max, y_min:y_max]
        y_core = self.y[:, x_min:x_max, y_min:y_max]

        return (x_core, y_core)


if __name__ == "__main__":
    STData = SpatiotemporalDataset("data/processed/npz", "1700_3100_13_13N", 5)

    STData.__getitem__(1030)
