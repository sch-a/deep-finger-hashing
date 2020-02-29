from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
import numpy as np
import datetime
import argparse

# Deep Autoencoder
from keras.optimizers import Adam

from read_data import get_dataset
from show_plots import show_loss_plot, show_reconstruction_plot


def create_autoencoder(input_shape, encoding_dim):

    input_img = Input(shape=(input_shape,))

    # "encoded" is the encoded representation of the inputs
    encoded = Dense(encoding_dim * 4, activation='relu')(input_img)
    encoded = Dense(encoding_dim * 2, activation='relu')(encoded)
    encoded = Dense(encoding_dim, activation='relu')(encoded)
    encoded = Lambda(lambda encoded: K.sign(encoded), name='sign_layer')(encoded)

    # "decoded" is the lossy reconstruction of the input
    decoded = Dense(encoding_dim * 2, activation='relu')(encoded)
    decoded = Dense(encoding_dim * 4, activation='relu')(decoded)
    decoded = Dense(input_shape, activation='sigmoid')(decoded)

    # this model maps an input to its reconstruction
    autoencoder = Model(input_img, decoded)

    # Separate Encoder model

    # this model maps an input to its encoded representation
    encoder = Model(input_img, encoded)

    # Separate Decoder model

    # create a placeholder for an encoded (32-dimensional) input
    encoded_input = Input(shape=(encoding_dim,))
    decoder_layer1 = autoencoder.layers[-3]
    decoder_layer2 = autoencoder.layers[-2]
    decoder_layer3 = autoencoder.layers[-1]
    # create the decoder model
    decoder = Model(encoded_input, decoder_layer3(decoder_layer2(decoder_layer1(encoded_input))))
    autoencoder.summary()

    autoencoder.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    return autoencoder, encoder, decoder


def check_and_get(arg, def_val):
    if arg:
        return arg
    else:
        return def_val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--x_size", "-x", type=int,
                        help="set the width of used images, 0 to use full width, else it will be cropped to center")
    parser.add_argument("--y_size", "-y", type=int,
                        help="set the height of used images, 0 to use full height, else it will be cropped to center")
    parser.add_argument("--epochs", "-e", type=int, help="number of epochs to train the model")
    parser.add_argument("--classes", "-c", type=int,
                        help="number of unique classes to use, 0 to use all, else they will be randomly picked")
    parser.add_argument("--dataset", "-d", type=str, help="the dataset to use, supports soco, fvc_db1, fvc_db3")
    parser.add_argument("--enc_dim", "-n", type=int, help="dimension of the encoded image")
    args = parser.parse_args()
    x_size = check_and_get(args.x_size, 0)
    y_size = check_and_get(args.y_size, 0)
    epochs = check_and_get(args.epochs, 100)
    classes = check_and_get(args.classes, 0)
    dataset_name = check_and_get(args.dataset, "soco")
    encoding_dim = check_and_get(args.enc_dim, 128)

    x_train, x_test, labels, new_x, new_y = get_dataset(dataset_name, classes, x_size, y_size)
    input_shape = new_x * new_y
    autoencoder, encoder, decoder = create_autoencoder(input_shape, encoding_dim)

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    # Train autoencoder for my_epochs epochs
    history = autoencoder.fit(x_train, x_train, epochs=epochs, batch_size=64, shuffle=True,
                              validation_data=(x_test, x_test),
                              verbose=2)

    # save model to file
    filepath = "autoencoder_model_{}_dims_{}_epochs_{}_input_shape_{}.h5".format(encoding_dim, epochs, input_shape,
                                                                                 datetime.datetime.now().strftime(
                                                                                     '%Y-%m-%d_%H-%M-%S'))
    autoencoder.save(filepath)
    # Visualize the reconstructed encoded representations
    # encode and decode some fingerprints
    # note that we take them from the *test* set
    encoded_imgs = encoder.predict(x_test)
    decoded_imgs = decoder.predict(encoded_imgs)

    show_loss_plot(history)
    show_reconstruction_plot(new_x, new_y, x_test, decoded_imgs, labels)

    K.clear_session()


if __name__ == '__main__':
    main()
