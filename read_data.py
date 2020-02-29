import glob

from tqdm import tqdm
import cv2
import numpy as np

from gen_labels import filter_n_unique_labels, split_perc, make_labels_from_file_list


def get_dataset(dataset_name, number_of_classes, x_dim, y_dim):
    file_list = get_file_list(dataset_name)
    np.random.shuffle(file_list)
    if number_of_classes > 0:
        file_list = filter_n_unique_labels(file_list, number_of_classes, dataset_name)
    x_train_files, x_test_files = split_perc(file_list, np.array([80, 20]))
    x_train, new_x, new_y = get_resized_images(x_train_files, x_dim, y_dim)
    x_test, _, _ = get_resized_images(x_test_files, x_dim, y_dim)
    labels, _ = make_labels_from_file_list(x_test_files, dataset_name)
    return x_train, x_test, labels, new_x, new_y


def get_file_list(dataset_name):
    if "soco" in dataset_name:
        return glob.glob(r"datasets\SOCOFing\Real\*.BMP") \
               + glob.glob(r"datasets\SOCOFing\Altered\Altered-Easy\*.BMP") \
               + glob.glob(r"datasets\SOCOFing\Altered\Altered-Medium\*.BMP") \
               + glob.glob(r"datasets\SOCOFing\Altered\Altered-Hard\*.BMP")
    elif "fvc" in dataset_name:
        if dataset_name.endswith("db1"):
            return glob.glob(r"datasets\FVC200X\DB1_B\*.TIF")
        elif dataset_name.endswith("db3"):
            return glob.glob(r"datasets\FVC200X\DB3_B\*.TIF")

    raise ValueError("invalid database")


def get_resized_images(file_list, x_dim, y_dim):
    img_list = []
    sample = cv2.imread(file_list[0], cv2.IMREAD_GRAYSCALE)
    base_y, base_x = sample.shape
    new_y, new_x = base_y, base_x
    if 0 < x_dim < base_x:
        new_x = x_dim
    if 0 < y_dim < base_y:
        new_y = y_dim

    print("getting image files...")
    for img_path in tqdm(file_list):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if new_y != base_y | new_x != base_x:
            img_list.append(crop_center(img, new_x, new_y).flatten())
        else:
            img_list.append(img.flatten())
    return np.array(img_list), new_x, new_y


def crop_center(img, cropx, cropy):
    y, x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy, startx:startx+cropx]
