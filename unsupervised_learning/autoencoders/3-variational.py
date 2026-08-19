#!/usr/bin/env python3
"""
Variational Autoencoder Module
"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a variational autoencoder

    Parameters:
    - input_dims: integer containing dimensions of model input
    - hidden_layers: list containing number of nodes for each hidden layer
    - latent_dims: integer containing dimensions of latent space

    Returns:
    - encoder: the encoder model
    - decoder: the decoder model
    - auto: the full autoencoder model
    """
    # Sampling function for reparameterization trick
    def sampling(args):
        """Samples latent vector z using mu and log_sig"""
        mu, log_sig = args
        epsilon = keras.backend.random_normal(
            shape=(keras.backend.shape(mu)[0], latent_dims)
        )
        return mu + keras.backend.exp(log_sig / 2) * epsilon

    # Encoder
    inputs = keras.Input(shape=(input_dims,))
    encoded = inputs
    for nodes in hidden_layers:
        encoded = keras.layers.Dense(nodes, activation='relu')(encoded)

    z_mean = keras.layers.Dense(latent_dims, activation=None)(encoded)
    z_log_sigma = keras.layers.Dense(latent_dims, activation=None)(encoded)

    z = keras.layers.Lambda(sampling)([z_mean, z_log_sigma])
    encoder = keras.Model(
        inputs, [z, z_mean, z_log_sigma], name='encoder'
    )

    # Decoder
    latent_inputs = keras.Input(shape=(latent_dims,))
    decoded = latent_inputs
    for nodes in reversed(hidden_layers):
        decoded = keras.layers.Dense(nodes, activation='relu')(decoded)
    decoded_output = keras.layers.Dense(
        input_dims, activation='sigmoid')(decoded)
    decoder = keras.Model(latent_inputs, decoded_output, name='decoder')

    # Full VAE Autoencoder
    auto_output = decoder(encoder(inputs)[0])
    auto = keras.Model(inputs, auto_output, name='autoencoder')

    # Custom VAE Loss
    def vae_loss(x, x_decoded_mean):
        """Calculates combined Reconstruction + KL Loss"""
        recon_loss = keras.losses.binary_crossentropy(x, x_decoded_mean)
        recon_loss *= input_dims
        kl_loss = -0.5 * keras.backend.sum(
            1 + z_log_sigma - keras.backend.square(z_mean) -
            keras.backend.exp(z_log_sigma),
            axis=-1
        )
        return keras.backend.mean(recon_loss + kl_loss)

    auto.compile(optimizer='adam', loss=vae_loss)

    return encoder, decoder, auto
