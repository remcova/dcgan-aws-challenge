import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Conv2DTranspose,
    Dense,
    Dropout,
    Flatten,
    Input,
    LayerNormalization,
    LeakyReLU,
    Reshape,
)
from tensorflow.keras.models import Model
import tensorflow_addons as tfa


def generator(latent_dim: int = 256, norm: str = "instance_norm", up_samplings: int = 5) -> Model:
    """
    Generator
    """
    Normalization = _get_norm_layer(norm)

    if norm == "instance_norm":
        Normalization(
            axis=3,
            center=True,
            scale=True,
            beta_initializer="random_uniform",
            gamma_initializer="random_uniform",
        )
    elif norm == "batch_norm":
        Normalization(momentum=0.8)

    model = tf.keras.Sequential()

    # foundation for 4x4 image
    n_nodes = 128 * 4 * 4
    model.add(Dense(n_nodes, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((4, 4, 128)))

    for _ in range(up_samplings):
        # upsample
        model.add(Conv2DTranspose(256, kernel_size=3, strides=2, padding="same"))
        model.add(Normalization())
        model.add(LeakyReLU(alpha=0.2))

    # last upsample to 256x256
    model.add(Conv2DTranspose(512, kernel_size=3, strides=2, padding="same"))
    model.add(Normalization())
    model.add(LeakyReLU(alpha=0.2))

    # output
    model.add(Conv2D(3, kernel_size=3, activation="tanh", padding="same"))

    model.summary()

    # Input
    noise = Input(shape=(latent_dim,))

    # Generated image
    img = model(noise)

    return Model(noise, img)


def discriminator(input_shape: tuple = (256, 256, 3), down_samplings: int = 5, use_hpu: bool = True) -> Model:
    """
    Discriminator
    """
    model = tf.keras.Sequential()
    if use_hpu:
        data_type = "float32"
    else:
        data_type = "float16"

    print(f"Global Policy : {tf.keras.mixed_precision.global_policy()}")

    model.add(
        Conv2D(
            128,
            kernel_size=3,
            padding="same",
            input_shape=input_shape,
        )
    )
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3, dtype=data_type))

    for _ in range(down_samplings):
        # downsample
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.3, dtype=data_type))

    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4, dtype=data_type))

    # output
    model.add(Flatten())
    model.add(Dense(1, activation="sigmoid"))

    model.summary()

    input = Input(shape=input_shape, dtype="float32")

    output = model(input)

    return Model(input, output)


def _get_norm_layer(norm):
    if norm == "batch_norm" or norm == None:
        return BatchNormalization
    elif norm == "instance_norm":
        # Experimental results show that instance normalization performs well on
        # style transfer when replacing batch normalization.
        # Recently, instance normalization has also been used as a replacement for
        # batch normalization in GANs.
        return tfa.layers.InstanceNormalization
    elif norm == "layer_norm":
        return LayerNormalization
