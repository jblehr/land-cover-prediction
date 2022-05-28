import numpy as np
from numpy import ndarray
from skimage import io
from matplotlib import pyplot as plt



def mask2rgb(mask):
    """
    Convert numpy mask to rbg thanks to color_label_dict
    :param mask: Numpy with labels value between 0 and 8
    :return: numpy array
    """
    color_label_dict = {0: [96, 96, 96],
                        1: [204, 204, 0],
                        2: [0, 204, 0],
                        3: [0, 0, 153],
                        4: [153, 76, 0],
                        5: [128, 0, 128],
                        6: [138, 178, 198]}

    assert mask.shape == (1024, 1024, 7)

    maskRGB = np.empty((mask.shape[0], mask.shape[1], 3)) #3 for RGB

    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            for k in range(mask.shape[2]):
                # TODO: Only getting first label; in some cases we have duplicates
                if mask[i,j,k] == 255:
                    maskRGB[i,j,:] = color_label_dict[k]
                    break
    return maskRGB.astype(np.uint8)


def plot_label_mask(label_array):
    '''
    Plot 7-d numpy array with all labels
    '''
    label_rgb = mask2rgb(label_array)
    plt.imshow(label_rgb)




def plot_normalized_image(image_path):
    """
    Read planet image and normalize data
    :param image_path: semi-supervised labels file path (.tif)
    :return: normalized numpy arrray of image
    """
    image = io.imread(image_path)
    new_min = -1
    new_max = 1
    image_normalized = (image - np.min(image)) / (
        np.max(image) - np.min(image)) * (new_max - new_min) + new_min
    return image_normalized


# Label dataset info
def get_info(dataset):

    print("Number of bands:", dataset.count)
    print("Pixels width:", dataset.width)
    print("Pixels height:", dataset.height)
    print("Bounds:", dataset.bounds)
    print("CRS:", dataset.crs)
    print("NUMPY shape:", dataset.read().shape)
    print("NUMPY shape of one band:", dataset.read(1).shape)

def summarize_classes(dataset):
    for class_idx in range(dataset.count):
        print(f'''Class index: {class_idx}. \
            Type: {dataset.dtypes[class_idx]}. \
            Max: {np.max(dataset.read(class_idx+1))}. \
            Min: {np.min(dataset.read(class_idx+1))}. \
            Med: {np.median(dataset.read(class_idx+1))}''')