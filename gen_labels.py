import numpy as np
from tqdm import tqdm


def make_labels_from_file_list(file_list, dataset):
    label_col = []
    num_label_col = []
    loc_label_map = dict()
    print("generating labels...")
    for file in tqdm(file_list):
        label = make_label_from_file(file, dataset)
        label_col.append(label)
        if label not in loc_label_map.keys():
            loc_label_map[label] = len(loc_label_map)
        num_label_col.append(loc_label_map.get(label))

    return label_col, num_label_col


def make_label_from_file(file, dataset):
    if "soco" in dataset:
        parts = file.split("\\")[-1].split("_")
        return (parts[0] + "_" + parts[3] + "_" + parts[4]).casefold()
    else:
        parts = file.split("\\")[-1].split("_")
        return (parts[0]).casefold()


def split_perc(l, perc):
    splits = np.cumsum(perc)/100.
    if splits[-1] != 1:
        raise ValueError("percents don't add up to 100")
    splits = splits[:-1]
    splits *= len(l)
    splits = splits.round().astype(np.int)

    return np.split(l, splits)


def filter_n_unique_labels(file_list, n, dataset):
    unique_labels = []
    unique_labels_files = []
    for file in file_list:
        if len(unique_labels) >= n:
            break
        else:
            label = make_label_from_file(file, dataset)
            if label not in unique_labels:
                unique_labels.append(label)
    for file in file_list:
        label = make_label_from_file(file, dataset)
        if label in unique_labels:
            unique_labels_files.append(file)
    return unique_labels_files
