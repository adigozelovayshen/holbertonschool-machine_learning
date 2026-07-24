#!/usr/bin/env python3
"""
ResNet-50 Architecture Module
"""
from tensorflow import keras as K

identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """
    Builds the ResNet-50 architecture as described in
    Deep Residual Learning for Image Recognition (2015)

    Returns:
    - The Keras model
    """
    initializer = K.initializers.HeNormal(seed=0)

    # Input tensor
    X = K.Input(shape=(224, 224, 3))

    # Stage 1: Conv -> BN -> ReLU -> MaxPool
    conv1 = K.layers.Conv2D(
        filters=64,
        kernel_size=(7, 7),
        strides=(2, 2),
        padding='same',
        kernel_initializer=initializer
    )(X)
    bn1 = K.layers.BatchNormalization(axis=-1)(conv1)
    act1 = K.layers.Activation('relu')(bn1)
    pool1 = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding='same'
    )(act1)

    # Stage 2: 1 Projection Block + 2 Identity Blocks
    X_stage2 = projection_block(pool1, [64, 64, 256], s=1)
    X_stage2 = identity_block(X_stage2, [64, 64, 256])
    X_stage2 = identity_block(X_stage2, [64, 64, 256])

    # Stage 3: 1 Projection Block + 3 Identity Blocks
    X_stage3 = projection_block(X_stage2, [128, 128, 512], s=2)
    X_stage3 = identity_block(X_stage3, [128, 128, 512])
    X_stage3 = identity_block(X_stage3, [128, 128, 512])
    X_stage3 = identity_block(X_stage3, [128, 128, 512])

    # Stage 4: 1 Projection Block + 5 Identity Blocks
    X_stage4 = projection_block(X_stage3, [256, 256, 1024], s=2)
    X_stage4 = identity_block(X_stage4, [256, 256, 1024])
    X_stage4 = identity_block(X_stage4, [256, 256, 1024])
    X_stage4 = identity_block(X_stage4, [256, 256, 1024])
    X_stage4 = identity_block(X_stage4, [256, 256, 1024])
    X_stage4 = identity_block(X_stage4, [256, 256, 1024])

    # Stage 5: 1 Projection Block + 2 Identity Blocks
    X_stage5 = projection_block(X_stage4, [512, 512, 2048], s=2)
    X_stage5 = identity_block(X_stage5, [512, 512, 2048])
    X_stage5 = identity_block(X_stage5, [512, 512, 2048])

    # Average Pooling
    avg_pool = K.layers.AveragePooling2D(
        pool_size=(7, 7),
        strides=(1, 1)
    )(X_stage5)

    # Dense Output Layer
    outputs = K.layers.Dense(
        units=1000,
        activation='softmax',
        kernel_initializer=initializer
    )(avg_pool)

    model = K.models.Model(inputs=X, outputs=outputs)
    return model
