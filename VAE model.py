import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape, Conv2D, Conv2DTranspose, BatchNormalization, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras import backend as K

# Define the VAE model
class VAE(tf.keras.Model):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        # Encoder with BatchNormalization and LeakyReLU
        self.encoder = tf.keras.Sequential([
            Input(shape=(64, 64, 3)),
            Conv2D(32, 3, strides=2, padding='same'),
            BatchNormalization(),
            LeakyReLU(alpha=0.2),
            Conv2D(64, 3, strides=2, padding='same'),
            BatchNormalization(),
            LeakyReLU(alpha=0.2),
            Flatten(),
            Dense(latent_dim + latent_dim)  # Mean and log variance
        ])

        # Decoder with BatchNormalization and LeakyReLU
        self.decoder = tf.keras.Sequential([
            Input(shape=(latent_dim,)),
            Dense(16 * 16 * 64, activation='relu'),
            Reshape((16, 16, 64)),
            Conv2DTranspose(64, 3, strides=2, padding='same'),
            BatchNormalization(),
            LeakyReLU(alpha=0.2),
            Conv2DTranspose(32, 3, strides=2, padding='same'),
            BatchNormalization(),
            LeakyReLU(alpha=0.2),
            Conv2DTranspose(3, 3, activation='sigmoid', padding='same')
        ])

    def call(self, inputs):
        mean, log_var = tf.split(self.encoder(inputs), num_or_size_splits=2, axis=1)
        z = self.reparameterize(mean, log_var)
        reconstruction = self.decoder(z)
        return reconstruction, mean, log_var

    def reparameterize(self, mean, log_var):
        epsilon = tf.random.normal(shape=mean.shape)
        return mean + tf.exp(0.5 * log_var) * epsilon

    def compute_loss(self, x, reconstruction, mean, log_var):
        reconstruction_loss = binary_crossentropy(K.flatten(x), K.flatten(reconstruction))
        kl_loss = -0.5 * tf.reduce_mean(log_var - tf.square(mean) - tf.exp(log_var) + 1)
        return tf.reduce_mean(reconstruction_loss + kl_loss)

# Prepare data with enhanced preprocessing
def preprocess_image(image):
    image = tf.image.resize(image, (64, 64))
    image = image / 255.0  # Normalize to [0, 1] range
    return image

def load_data():
    (x_train, _), (x_test, _) = tf.keras.datasets.cifar10.load_data()
    x_train = np.array([preprocess_image(img) for img in x_train])
    x_test = np.array([preprocess_image(img) for img in x_test])
    return x_train, x_test

# Train the VAE
def train_vae():
    x_train, x_test = load_data()
    latent_dim = 16  # Increased latent dimension for better representation
    vae = VAE(latent_dim)

    # Reduce the learning rate for better convergence
    vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005))

    # Custom training loop
    @tf.function
    def train_step(x):
        with tf.GradientTape() as tape:
            reconstruction, mean, log_var = vae(x, training=True)
            loss = vae.compute_loss(x, reconstruction, mean, log_var)
        gradients = tape.gradient(loss, vae.trainable_variables)
        vae.optimizer.apply_gradients(zip(gradients, vae.trainable_variables))
        return loss

    for epoch in range(20):  # Increased epochs for more training
        epoch_loss = 0
        for i in range(0, len(x_train), 32):
            batch = x_train[i:i+32]
            loss = train_step(batch)
            epoch_loss += loss
        print(f"Epoch {epoch+1}, Loss: {epoch_loss.numpy()}")

    # Save the model weights after training
    vae.save_weights("vae_weights_updated.weights.h5")
    print("Model weights saved.")

    # Evaluation on test data
    test_loss = 0
    for i in range(0, len(x_test), 32):
        batch = x_test[i:i+32]
        reconstruction, mean, log_var = vae(batch, training=False)
        loss = vae.compute_loss(batch, reconstruction, mean, log_var)
        test_loss += loss
    print(f"Evaluation Loss: {test_loss.numpy() / len(x_test)}")

# Run training
train_vae()
