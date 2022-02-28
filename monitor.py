import os
import matplotlib.pyplot as plt
import numpy as np


def plot_losses(d_loss_log_real, d_loss_log_fake, g_loss_log_adv):
    print("Plotting Discriminator and Generator Loss Logs")
    plt.plot(
        d_loss_log_real[:, 0],
        d_loss_log_real[:, 1],
        label="Discriminator Loss - Real",
    )
    plt.plot(
        d_loss_log_fake[:, 0],
        d_loss_log_fake[:, 1],
        label="Discriminator Loss - Fake",
    )
    plt.plot(g_loss_log_adv[:, 0], g_loss_log_adv[:, 1], label="Generator Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("DCGAN")
    plt.grid(True)
    plt.show()


def preview_dataset(data: list):
    """
    Preview dataset images
    :param data: Given training set
    """
    print(f"Cataract Image Set")

    _, ax = plt.subplots(5, 5, figsize=(15, 15))

    for i, data in enumerate(data[:25]):
        img_data = data[0]
        ax[i // 5, i % 5].imshow(img_data)
        ax[i // 5, i % 5].axis("off")

    plt.show()


def monitor_generated_samples(
    samples_num: int,
    img_shape: tuple,
    samples_dir: str,
    gen_imgs: np.array,
    epoch_num: int,
):
    """
    Show & Save Generated Samples
    :param epoch: Amount of epochs. Used to save in filename for the generated sample.
    """
    x_fake = gen_imgs

    for k in range(samples_num):
        plt.subplot(2, 5, k + 1)
        plt.imshow(np.uint8(255 * (x_fake[k].reshape(img_shape))))
        plt.savefig(os.path.join(samples_dir, f"{epoch_num}_samples.png"))
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()
    plt.show()
