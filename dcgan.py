import os
import glob
import pathlib
import time
from datetime import datetime
from random import shuffle

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import opendatasets as od
import pandas as pd
import tensorflow as tf
import tensorflow.experimental.numpy as tnp
import tensorflow_addons as tfa

from habana_frameworks.tensorflow.ops.instance_norm import HabanaInstanceNormalization

from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    BatchNormalization,
    LayerNormalization,
    Conv2D,
    Conv2DTranspose,
    Dense,
    Dropout,
    Flatten,
    Input,
    LeakyReLU,
    Reshape,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

# Enable numpy behavior for TF
tnp.experimental_enable_numpy_behavior()


class DCGAN:
    def __init__(self, use_hpu=True, use_horovod=True):
        # HPU
        self.use_hpu = use_hpu
        # Horovod
        self.use_horovod = use_horovod

        # Hyperparameters
        self.epochs = 30000
        self.batch_size = 12
        self.latent_dim = 256

        self.img_size = 256
        self.img_shape = (self.img_size, self.img_size, 3)

        # Data type
        self.data_type = np.float32

        # Required models for GAN
        self.disc = None
        self.gen = None
        self.combined = None

        # Save interval for generated samples
        self.save_interval = 10
        self.samples_am = 10

        # Use latest model checkpoint
        self.use_checkpoint = False

        # Save interval for model checkpoint
        self.save_checkpoint_interval = 25

        # Get date and time of current run
        now = datetime.now()
        self.datetime = now.strftime("%d_%m_%Y_%H_%M_%S")

        # Create required folders
        (self.model_dir, self.checkpoint_dir, self.samples_dir) = self.create_run_folders()

    def is_master(self, use_hvd, horovod):
        return use_hvd is False or horovod.rank() == 0
        
    def is_local_master(self, use_hvd, horovod):
        return use_hvd is False or horovod.local_rank() == 0

    def is_not_worker_zero(self, use_hvd, horovod):
        if use_hvd is True and horovod.rank() != 0:
            return False

    def create_run_folders(self):
        """
        Create required directories for saving results from this run
        """
        model_path = f"models/{self.datetime}"
        if not os.path.exists(model_path):
            os.mkdir(model_path)
            model_dir = os.path.join(model_path)
        else:
            model_dir = os.path.join(model_path)

        checkpoint_path = f"checkpoints/{self.datetime}"
        if not os.path.exists(checkpoint_path):
            os.mkdir(checkpoint_path)
            checkpoint_dir = os.path.join(checkpoint_path)
        else:
            checkpoint_dir = os.path.join(checkpoint_path)

        samples_dir = f"generated_samples/{self.datetime}"
        if not os.path.exists(samples_dir):
            os.mkdir(samples_dir)
            samples_dir = os.path.join(samples_dir)
        else:
            samples_dir = os.path.join(samples_dir)

        return [model_dir, checkpoint_dir, samples_dir]

    def download_data(self):
        """
        Download Required Data
        """
        od.download(
            "https://www.kaggle.com/andrewmvd/ocular-disease-recognition-odir5k"
        )

    def process_data(self, data_image_list: list, data_folder: str) -> list:
        """
        Process Training Data
        :param data_image_list: Raw dataset
        :param data_folder: Path to training images
        :returns: Processed data
        """
        data_df = []
        for img in tqdm(data_image_list):
            path = os.path.join(data_folder, img)
            img = cv.imread(path)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            img = cv.resize(
                img, (self.img_size, self.img_size), interpolation=cv.INTER_NEAREST
            )
            data_df.append([np.array(img)])
        shuffle(data_df)
        return data_df

    def has_cataract(self, text) -> bool:
        """
        Checks if the word 'cataract' occurs in selected record from the data (.csv) file
        :param text: Text to screen for the word 'cataract'
        :returns: boolean
        """
        if "cataract" in text:
            return 1
        else:
            return 0

    def show_images(self, data: list):
        """
        Preview dataset images
        :param data: Given training set
        """
        print(f"Cataract Image Set")

        f, ax = plt.subplots(5, 5, figsize=(15, 15))

        for i, data in enumerate(data[:25]):
            img_data = data[0]
            ax[i // 5, i % 5].imshow(img_data)
            ax[i // 5, i % 5].axis("off")

        plt.show()

    def prepare_dataset(self) -> list:
        """
        Prepare dataset
        """
        training_data_folder = os.path.join(
            "./ocular-disease-recognition-odir5k/ODIR-5K/ODIR-5K/Training Images"
        )
        df = pd.read_csv(
            os.path.join("./ocular-disease-recognition-odir5k/", "full_df.csv")
        )

        df["left_cataract"] = df["Left-Diagnostic Keywords"].apply(
            lambda x: self.has_cataract(x)
        )
        df["right_cataract"] = df["Right-Diagnostic Keywords"].apply(
            lambda x: self.has_cataract(x)
        )

        left_cataract = df.loc[(df.C == 1) & (df.left_cataract == 1)][
            "Left-Fundus"
        ].values
        right_cataract = df.loc[(df.C == 1) & (df.right_cataract == 1)][
            "Right-Fundus"
        ].values

        print("Number of images in left cataract: {}".format(len(left_cataract)))
        print("Number of images in right cataract: {}".format(len(right_cataract)))

        cataract_list = np.concatenate((left_cataract, right_cataract), axis=0)

        print('Total "cataract" images : ', len(cataract_list))

        cat_df = self.process_data(cataract_list, training_data_folder)

        return cat_df

    def rescale_data(self, data_image_list: np.array) -> np.array:
        """
        Rescale Data
        :param data_image_list: Data to rescale
        :returns: Processed data_image_list
        """
        min_max = tf.keras.layers.experimental.preprocessing.Rescaling(
            1.0 / 127.5, offset=-1
        )
        processed_data = min_max(data_image_list)

        return processed_data

    def create_dataset(self, horovod = None) -> np.array:
        # Download Data
        self.download_data()

        # Create Training Set
        train = self.prepare_dataset()
        shuffle(train)

        # Visualize Training Set
        self.show_images(train)

        def create_training_set():
            training_set = (
                np.array([i[0] for i in train])
                .astype(self.data_type)
                .reshape(-1, self.img_size, self.img_size, 3)
            )
            return training_set

        # Ensure only 1 process downloads the data on each node
        if horovod is not None:
            if horovod.local_rank() == 0:
                X = create_training_set()
                horovod.broadcast(0, 0)
            else:
                horovod.broadcast(0, 0)
                X = create_training_set()
        else:
            X = create_training_set()

        # Normalize Training Data (MinMax Scaling)
        X = self.rescale_data(X)

        # Return created dataset
        return X

    def generator(self, norm: str = "instance_norm", up_samplings: int = 5) -> Model:
        """
        Generator
        """
        Normalization = self._get_norm_layer(norm)

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
        model.add(Dense(n_nodes, input_dim=self.latent_dim))
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
        noise = Input(shape=(self.latent_dim,))

        # Generated image
        img = model(noise)

        return Model(noise, img)

    def discriminator(self, down_samplings: int = 5) -> Model:
        """
        Discriminator
        """
        model = tf.keras.Sequential()

        print(f"Global Policy : {tf.keras.mixed_precision.global_policy()}")

        model.add(
            Conv2D(
                128,
                kernel_size=3,
                padding="same",
                input_shape=(self.img_size, self.img_size, 3),
            )
        )
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.3, dtype="float32"))

        for _ in range(down_samplings):
            # downsample
            model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
            model.add(LeakyReLU(alpha=0.2))
            model.add(Dropout(0.3, dtype="float32"))

        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4, dtype="float32"))

        # output
        model.add(Flatten())
        model.add(Dense(1, activation="sigmoid"))

        model.summary()

        input = Input(shape=(self.img_size, self.img_size, 3), dtype="float32")

        output = model(input)

        return Model(input, output)

    def train(
        self,
        data: np.array,
        epochs: int = 1000,
        batch_size: int = 12,
        save_interval: int = 10,
        checkpoint: tf.train.Checkpoint = None,
        checkpoint_prefix: str = "ckpt",
        verbose: bool = False,
        horovod=None, 
    ):
        """
        Training Loop
        :param data: Given training set
        :param epochs: Amount of epochs to run
        :param batch_size: Batch Size
        :param save_interval: Used as interval to save generated samples
        :param checkpoint: Model checkpointing
        :param checkpoint_prefix: Checkpoint file prefix
        :param horovod: Horovod init
        """
        if horovod is not None:
            # Horovod: broadcast initial variable states from rank0 to all other processes.
            # This is necessary to ensure consistent initialization of all workers when
            # training is started with random weights or restored from a checkpoint.
            horovod.callbacks.BroadcastGlobalVariablesCallback(0)

        # Load the dataset
        X_train = np.stack(data, axis=0)

        # Define a half batch size for the discriminator
        # First half for real images, second half for fake images.
        half_batch = int(batch_size / 2)

        # Array init for loss logging
        d_loss_logs_r = []
        d_loss_logs_f = []
        g_loss_logs = []

        # Adverserial ground truths
        valid = np.ones((half_batch, 1))
        fake = np.zeros((half_batch, 1))

        for epoch in range(epochs):
            # Record time for current epoch
            start = time.time()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of real images
            idx = np.random.randint(0, X_train.shape[0], half_batch)
            imgs = X_train[idx]

            # Generate noise by using a uniform distribution
            # Additionally, the generator uses the hyperbolic tangent (tanh) activation function
            # in the output layer, and inputs to the generator & discriminator are scaled to the range [-1, 1].

            noise = np.random.uniform(-1, 1, size=[half_batch, self.latent_dim])

            # Generate a batch of fake images
            gen_imgs = self.gen.predict_on_batch(noise)

            # Train the discriminator on real and fake images, separately
            # Research showed that separate training is more effective.

            self.disc.trainable = True

            d_loss_real = self.disc.train_on_batch(imgs, valid)
            d_loss_fake = self.disc.train_on_batch(gen_imgs, fake)

            # Take average loss from real and fake images.
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------
            # Within the same loop we train the Generator, by setting the input noise and
            # ultimately training the Generator to have the Discriminator label its samples as valid

            # Create as many noise vectors as defined by the batch size.
            # Based on uniform distribution. Output will be of size (batch size, latent_dim)
            noise = np.random.uniform(-1, 1, size=[batch_size, self.latent_dim])

            # The generator wants the discriminator to label the generated samples as valid (ones)
            # This is where the generator is trying to trick the discriminator into believing
            # that the generated image is true
            valid_y = np.array([1] * batch_size)

            # Generator is part of combined where it got directly linked with the discriminator
            # Train the generator with noise as x and 1 as y.
            # Again, 1 as the output as it is adversarial and if generator did a great
            # job of fooling the discriminator, then the output would be 1 (true)
            g_loss = self.combined.train_on_batch(noise, valid_y)

            # Plot the progress
            accuracy = 100 * d_loss[1]
            if verbose == False:
                print(
                    f"Epoch: {epoch}, [d_avg_loss: {d_loss[0]}, d_loss_fake: {d_loss_fake[0]}, \
                        d_loss_real: {d_loss_real[0]}, acc.: {accuracy}] [generator Loss: {g_loss}]"
                )

            # Store the losses
            d_loss_logs_r.append([epoch, d_loss[0]])
            d_loss_logs_f.append([epoch, d_loss[1]])
            g_loss_logs.append([epoch, g_loss])

            # If at save interval -> save generated image samples
            if epoch % save_interval == 0:
                self.show_samples(epoch)

            # Save checkpoint every X epoch
            if epoch % self.save_checkpoint_interval == 0:
                checkpoint.save(file_prefix=checkpoint_prefix)

            print(f"Time for epoch {epoch + 1} is {time.time()-start} sec")

        d_loss_logs_r_a = np.array(d_loss_logs_r)
        d_loss_logs_f_a = np.array(d_loss_logs_f)
        g_loss_logs_a = np.array(g_loss_logs)

        # At the end of training plot the losses vs epochs
        plt.plot(
            d_loss_logs_r_a[:, 0],
            d_loss_logs_r_a[:, 1],
            label="Discriminator Loss - Real",
        )
        plt.plot(
            d_loss_logs_f_a[:, 0],
            d_loss_logs_f_a[:, 1],
            label="Discriminator Loss - Fake",
        )
        plt.plot(g_loss_logs_a[:, 0], g_loss_logs_a[:, 1], label="Generator Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("DCGAN")
        plt.grid(True)
        plt.show()

    def show_samples(self, epoch: int):
        """
        Show Samples
        :param epoch: Amount of epochs. Used to save in filename for the generated sample.
        """
        noise = np.random.uniform(-1, 1, size=[self.samples_am, self.latent_dim])
        x_fake = self.gen.predict(noise)

        for k in range(self.samples_am):
            plt.subplot(2, 5, k + 1)
            plt.imshow(
                np.uint8(255 * (x_fake[k].reshape(self.img_size, self.img_size, 3)))
            )
            plt.savefig(os.path.join(self.samples_dir, f"{epoch}_samples.png"))
            plt.xticks([])
            plt.yticks([])

        plt.tight_layout()
        plt.show()

    def run(self):
        """
        Run DCGAN
        """
        # Horovod init
        horovod = None
        if self.use_horovod:
            import horovod.tensorflow.keras as horovod
            horovod.init()

            # Adjust batch size dynamically by the available Horovod Size
            self.batch_size = self.batch_size * horovod.size()

        # Create required folders
        if horovod.rank() != 0:
            self.create_run_folders()
            
        # Create dataset
        X = self.create_dataset(horovod=horovod)

        # Change data type if HPU is used
        if self.use_hpu:
            self.data_type = tf.float32.as_numpy_dtype

        # Setup GAN Optimizers
        generator_optimizer = Adam(2e-4, beta_1=0.5)
        discriminator_optimizer = Adam(2e-4, beta_1=0.5)

        # Setup Horovod Optimizer
        if horovod is not None:
            optimizer = Adam(2e-4 * horovod.size())
            if horovod.size() > 1:
                optimizer = horovod.DistributedOptimizer(optimizer)
        else:
            optimizer = Adam(2e-4, beta_1=0.5)

        # Build and compile the discriminator first.
        # Generator will be trained as part of the combined model later on.
        self.disc = self.discriminator()
        self.disc.compile(
            loss="binary_crossentropy",
            optimizer=discriminator_optimizer,
            metrics=["accuracy"],
        )

        # Since we are only generating (faking) images, we do not track any metrics.
        self.gen = self.generator()
        self.gen.compile(loss="binary_crossentropy", optimizer=generator_optimizer)

        # This builds the Generator and defines the input noise.
        # In a GAN the Generator network takes noise Z as an input to produce its images.
        z = Input(shape=(self.latent_dim,))
        img = self.gen(z)

        # This ensures that when we combine our networks we only train the Generator.
        # While the generator is training, we do not want the discriminator weights to be adjusted.
        # This doesn't affect the above descriminator training.
        self.disc.trainable = False

        # Validity check on the generated image
        valid = self.disc(img)

        # Here we combine the models and also set our loss function and optimizer.
        self.combined = Model(z, valid)

        # Create checkpoint model for the adverserial network
        checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        checkpoint = tf.train.Checkpoint(
            generator_optimizer=generator_optimizer,
            discriminator_optimizer=discriminator_optimizer,
            generator=self.gen,
            discriminator=self.disc,
        )

        if self.use_checkpoint:
            # Get the latest checkpoint model
            self.latest_checkpoint_dir = max(
                glob.glob(os.path.join("checkpoints", "*/")), key=os.path.getmtime
            )
            latest = tf.train.latest_checkpoint(self.latest_checkpoint_dir)

            if latest != None:
                self.combined.load_weights(latest)

        # Compile Combined model
        self.combined.compile(loss="binary_crossentropy", optimizer=optimizer)

        print(f"Model is compiled, settings hooks")

        # Train the network
        self.train(
            data=X,
            epochs=self.epochs,
            batch_size=self.batch_size,
            save_interval=self.save_interval,
            checkpoint=checkpoint,
            checkpoint_prefix=checkpoint_prefix,
            horovod=horovod,
            verbose=self.is_not_worker_zero(self.use_horovod, horovod)
        )

        if self.is_local_master(self.use_horovod, horovod):
            # Save model for future use to generate synthetic data
            self.gen.save(os.path.join(self.model_dir, "output_model.h5"))

        # Release resources from GPU memory
        K.clear_session()


    def _get_norm_layer(self, norm):
        if norm == "none":
            return lambda: lambda x: x
        elif norm == "batch_norm":
            return BatchNormalization
        elif norm == "instance_norm":
            # Experimental results show that instance normalization performs well on
            # style transfer when replacing batch normalization.
            # Recently, instance normalization has also been used as a replacement for
            # batch normalization in GANs.
            return tfa.layers.InstanceNormalization
        elif norm == "layer_norm":
            return LayerNormalization

if __name__ == "__main__":
    # Run this file with the following command:
    # mpirun -np 8 python3 dcgan.py

    # Args
    use_hpu = True
    use_horovod = True

    if use_hpu:
        # Load Habana module
        from habana_frameworks.tensorflow import load_habana_module

        load_habana_module()

        # When set to True this routine will generate a log with the
        # device placement of all of the TensorFlow ops in the program.
        tf.debugging.set_log_device_placement(False)

        # Configure computing data type
        tf.keras.mixed_precision.set_global_policy(
            "mixed_bfloat16"
        )  # used for in the network architecture

        # Replace default TF Instance Normalization with Habana compatible Instance Normalization
        tfa.layers.InstanceNormalization = HabanaInstanceNormalization

    # Run DCGAN
    DCGAN(use_hpu=use_hpu, use_horovod=use_horovod).run()
