import matplotlib.pyplot as plt


def show_loss_plot(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def show_reconstruction_plot(x_shape, y_shape, x_test, decoded_imgs, labels):
    n = min(len(labels),  5) # how many digits we will display
    plt.figure(figsize=(2 * n, 2), dpi=100)
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i].reshape(y_shape, x_shape))
        plt.gray()
        ax.set_axis_off()

        # display reconstruction
        ax = plt.subplot(2, n, i + n + 1)
        plt.imshow(decoded_imgs[i].reshape(y_shape, x_shape))
        plt.gray()
        ax.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            right=False,
            left=False,
            labelleft=False,
            labelbottom=False)  # labels along the bottom edge are off
        ax.set_xlabel(labels[i])
    plt.show()


