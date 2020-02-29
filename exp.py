
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree, distance
from keras import Model, Input
from keras.layers import Dense, Lambda
from keras.models import load_model
from keras import backend as K

from gen_labels import filter_n_unique_labels, make_labels_from_file_list
from read_data import get_resized_images, get_file_list

full_file_list = get_file_list("soco")

encoding_dim = 128
input_img = Input(shape=(9888, ))
# autoencoder = load_model('autoencoder_model_2020-01-16_10-43-13.h5')
# autoencoder = load_model('autoencoder_model_128_dims_500_epochs_9888_input_shape_2020-02-10_14-31-01.h5')
# autoencoder = load_model('autoencoder_model_128_dims_1500_epochs_9888_input_shape_2020-02-17_08-13-06.h5')
# autoencoder = load_model('autoencoder_model_128_dims_1000_epochs_7998_input_shape_2020-02-25_11-02-22.h5')
# autoencoder = load_model('autoencoder_model_128_dims_1000_epochs_7998_input_shape_2020-02-25_11-41-42.h5')
autoencoder = load_model('autoencoder_model_128_dims_100_epochs_9888_input_shape_2020-02-27_22-36-09.h5')

encoded = Dense(encoding_dim * 4, activation='relu')(input_img)
encoded = Dense(encoding_dim * 2, activation='relu')(encoded)
encoded = Dense(encoding_dim, activation='relu')(encoded)
encoded = Lambda(lambda encoded: K.sign(encoded), name='sign_layer')(encoded)
encoder = Model(input_img, encoded)

encoded_input = Input(shape=(encoding_dim, ))
decoder_layer1 = autoencoder.layers[-3]
decoder_layer2 = autoencoder.layers[-2]
decoder_layer3 = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer3(decoder_layer2(decoder_layer1(encoded_input))))


def calc_hashes_for_files(file_list, classes):
    print("loading files")
    file_list = list(set(file_list))
    part_list = filter_n_unique_labels(file_list, classes, "soco")
    x_test, _, _ = get_resized_images(part_list, 0, 0)
    encoded_imgs = encoder.predict(x_test)
    hash_arr_list = encoded_imgs
    print("generated images", len(encoded_imgs))
    hashes = []
    print("calculating hashes...")
    bool_arrs = []
    for hash_arr in hash_arr_list:
        bool_arr = np.greater(hash_arr, np.zeros(128))
        bool_arrs.append(bool_arr)
        gen_hash = sum(2**i for i, v in enumerate(reversed(bool_arr)) if v)
        hashes.append(gen_hash)
    avg_hamming_dists = []
    sum_of_ones = np.zeros(128)
    for bool_arr in tqdm(bool_arrs):
        sum_of_ones = sum_of_ones + np.array(bool_arr).astype(int)
        dist_arr = []
        for other_bool_arr in tqdm(bool_arrs):
            if not np.array_equal(bool_arr, other_bool_arr):
                dist_arr.append(distance.hamming(bool_arr, other_bool_arr))
        avg_hamming_dists.append(np.average(dist_arr))
    ones_avg = np.divide(sum_of_ones, len(bool_arrs))
    print(len(hashes))
    print(len(set(hashes)))
    return hashes, avg_hamming_dists, ones_avg


def show_class_diff(file_list, classes):
    file_list = list(set(file_list))
    part_list = filter_n_unique_labels(file_list, classes, "soco")
    labels, _ = make_labels_from_file_list(part_list, "soco")
    x_test, _, _ = get_resized_images(part_list, 0, 0)
    encoded_imgs = encoder.predict(x_test)
    labeled_imgs = zip(labels, encoded_imgs, part_list)
    in_class = []
    out_class = []
    class_name = labels[0]
    for img in labeled_imgs:
        if img[0] == class_name:
            in_class.append(img)
        else:
            out_class.append(img)
    out_class.append(in_class[0])

    in_imgs = [i[1] for i in in_class]
    out_imgs = [i[1] for i in out_class]
    in_tree = KDTree(in_imgs)
    out_tree = KDTree(out_imgs)
    in_distances = in_tree.query(in_imgs, 2)
    out_distances = out_tree.query(out_imgs, 2)
    filtered_in_dist = np.delete(in_distances[0], 0, 1)
    filtered_out_dist = np.delete(out_distances[0], 0, 1)
    labeled_in_dist = list(zip(in_class, filtered_in_dist))
    labeled_out_dist = list(zip(out_class, filtered_out_dist))
    x_in = np.arange(len(labeled_in_dist))
    x_out = np.arange(len(labeled_out_dist))
    in_dst = [i[1][0] for i in labeled_in_dist]
    out_dst = [i[1][0] for i in labeled_out_dist]
    in_file_name = [i[0][2].split('\\')[-1] for i in labeled_in_dist]
    out_lbl = [i[0][0] for i in labeled_out_dist]
    plt.bar(x_in, in_dst)
    plt.xticks(x_in, in_file_name, rotation='vertical', fontsize=8)
    plt.show()
    plt.bar(x_out, out_dst)
    plt.xticks(x_out, out_lbl, rotation='vertical', fontsize=8)
    plt.show()


# show_class_diff(full_file_list, 10)
calc_hashes, calc_avg_hamming_dists, calc_avg_ones = calc_hashes_for_files(full_file_list, 100)
calc_hashes = np.array(calc_hashes, dtype=float)
n_bins = 500
# We can set the number of bins with the `bins` kwarg
plt.hist(calc_hashes, bins=n_bins)
plt.title('hash distribution')
plt.show()

# calc_avg_ones.sort()
plt.plot(calc_avg_ones)
plt.axhline(y=np.average(calc_avg_ones), color='r')
plt.axhline(y=0.5, color='g', linestyle='--')
plt.title('average 1/0')
plt.show()

calc_avg_hamming_dists.sort()
plt.plot(calc_avg_hamming_dists)
plt.title('average hamming distances')
plt.ylim(bottom=0)
plt.show()
