#!/usr/bin/env python3
"""
Bayesian Optimization with GPyOpt
"""
import GPyOpt
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks, layers, models, regularizers


def build_and_train_model(x_train, y_train, x_val, y_val, params):
    """
    Builds, trains and evaluates a Keras model given hyperparameters
    """
    lr = float(params[:, 0][0])
    num_units = int(params[:, 1][0])
    dropout = float(params[:, 2][0])
    l2_reg = float(params[:, 3][0])
    batch_size = int(params[:, 4][0])

    model = models.Sequential([
        layers.Dense(
            num_units,
            activation='relu',
            kernel_regularizer=regularizers.l2(l2_reg),
            input_shape=(x_train.shape[1],)
        ),
        layers.Dropout(dropout),
        layers.Dense(10, activation='softmax')
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    checkpoint_filename = (
        f"ckpt_lr{lr:.4f}_units{num_units}_drop{dropout:.2f}"
        f"_l2{l2_reg:.4f}_batch{batch_size}.h5"
    )

    my_callbacks = [
        callbacks.EarlyStopping(monitor='val_loss', patience=3),
        callbacks.ModelCheckpoint(
            filepath=checkpoint_filename,
            monitor='val_loss',
            save_best_only=True
        )
    ]

    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=15,
        batch_size=batch_size,
        callbacks=my_callbacks,
        verbose=0
    )

    val_loss = min(history.history['val_loss'])
    return val_loss


def main():
    """
    Main function to run Bayesian Optimization
    """
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_val, y_val) = mnist.load_data()
    x_train = x_train.reshape(-1, 28 * 28) / 255.0
    x_val = x_val.reshape(-1, 28 * 28) / 255.0

    domain = [
        {'name': 'learning_rate', 'type': 'continuous', 'domain': (1e-4, 1e-2)},
        {'name': 'num_units', 'type': 'discrete', 'domain': (32, 64, 128, 256)},
        {'name': 'dropout_rate', 'type': 'continuous', 'domain': (0.1, 0.5)},
        {'name': 'l2_reg', 'type': 'continuous', 'domain': (1e-5, 1e-2)},
        {'name': 'batch_size', 'type': 'discrete', 'domain': (32, 64, 128)}
    ]

    def fit_model(params):
        return build_and_train_model(x_train, y_train, x_val, y_val, params)

    optimizer = GPyOpt.methods.BayesianOptimization(
        f=fit_model,
        domain=domain,
        acquisition_type='EI',
        exact_feval=True
    )

    optimizer.run_optimization(max_iter=30)

    # Plot convergence
    optimizer.plot_convergence()
    plt.savefig('convergence_plot.png')

    # Save report
    with open('bayes_opt.txt', 'w') as f:
        f.write("Bayesian Optimization Report\n")
        f.write("============================\n")
        f.write(f"Best Hyperparameters (X_opt):\n")
        f.write(f"Learning Rate: {optimizer.x_opt[0]}\n")
        f.write(f"Num Units: {optimizer.x_opt[1]}\n")
        f.write(f"Dropout Rate: {optimizer.x_opt[2]}\n")
        f.write(f"L2 Regularization: {optimizer.x_opt[3]}\n")
        f.write(f"Batch Size: {optimizer.x_opt[4]}\n")
        f.write(f"Best Validation Loss (Y_opt): {optimizer.fx_opt}\n")


if __name__ == '__main__':
    main()
