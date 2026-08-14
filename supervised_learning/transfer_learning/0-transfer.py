#!/usr/bin/env python3
"""
Transfer Learning on CIFAR-10 using Keras Applications (DenseNet121)
"""
import tensorflow.keras as K


def preprocess_data(X, Y):
    """
    Preprocesses CIFAR-10 image data and labels for DenseNet121.
    """
    X_p = K.applications.densenet.preprocess_input(X)
    Y_p = K.utils.to_categorical(Y, 10)
    return X_p, Y_p


if __name__ == '__main__':
    (X_train, Y_train), (X_val, Y_val) = K.datasets.cifar10.load_data()

    X_train_p, Y_train_p = preprocess_data(X_train, Y_train)
    X_val_p, Y_val_p = preprocess_data(X_val, Y_val)

    inputs = K.Input(shape=(32, 32, 3))
    scaled_inputs = K.layers.Lambda(
        lambda img: K.backend.resize_images(img, 7, 7, "channels_last")
    )(inputs)

    base_model = K.applications.DenseNet121(
        include_top=False,
        weights='imagenet',
        input_tensor=scaled_inputs
    )

    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = K.layers.GlobalAveragePooling2D()(x)
    x = K.layers.BatchNormalization()(x)
    x = K.layers.Dense(256, activation='relu')(x)
    x = K.layers.Dropout(0.4)(x)
    outputs = K.layers.Dense(10, activation='softmax')(x)

    model = K.models.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=K.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        K.callbacks.ModelCheckpoint(
            filepath='cifar10.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        K.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.5,
            patience=2,
            verbose=1
        )
    ]

    model.fit(
        X_train_p,
        Y_train_p,
        validation_data=(X_val_p, Y_val_p),
        batch_size=64,
        epochs=10,
        callbacks=callbacks,
        verbose=1
    )

    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=K.optimizers.Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        X_train_p,
        Y_train_p,
        validation_data=(X_val_p, Y_val_p),
        batch_size=64,
        epochs=5,
        callbacks=callbacks,
        verbose=1
    )

    model.save('cifar10.h5')
