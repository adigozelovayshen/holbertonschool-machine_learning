#!/usr/bin/env python3
"""
Neural Style Transfer module with
Variational Cost and Image Generation
"""
import numpy as np
import tensorflow as tf


class NST:
    """
    Class NST that performs tasks
    for neural style transfer
    """
    style_layers = [
        'block1_conv1',
        'block2_conv1',
        'block3_conv1',
        'block4_conv1',
        'block5_conv1'
    ]
    content_layer = 'block5_conv2'

    def __init__(
        self, style_image, content_image,
        alpha=1e4, beta=1, var=10
    ):
        """
        Constructor for NST class.
        """
        if (not isinstance(style_image, np.ndarray) or
                len(style_image.shape) != 3 or
                style_image.shape[2] != 3):
            raise TypeError(
                "style_image must be a numpy.ndarray "
                "with shape (h, w, 3)"
            )

        if (not isinstance(content_image, np.ndarray) or
                len(content_image.shape) != 3 or
                content_image.shape[2] != 3):
            raise TypeError(
                "content_image must be a numpy.ndarray "
                "with shape (h, w, 3)"
            )

        if (not isinstance(alpha, (int, float, np.number)) or
                isinstance(alpha, bool) or alpha < 0):
            raise TypeError(
                "alpha must be a non-negative number"
            )

        if (not isinstance(beta, (int, float, np.number)) or
                isinstance(beta, bool) or beta < 0):
            raise TypeError(
                "beta must be a non-negative number"
            )

        if (not isinstance(var, (int, float, np.number)) or
                isinstance(var, bool) or var < 0):
            raise TypeError(
                "var must be a non-negative number"
            )

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.var = var
        self.load_model()
        self.generate_features()

    @staticmethod
    def scale_image(image):
        """
        Rescales an image such that its
        pixel values are between 0 and 1.
        """
        if (not isinstance(image, np.ndarray) or
                len(image.shape) != 3 or
                image.shape[2] != 3):
            raise TypeError(
                "image must be a numpy.ndarray "
                "with shape (h, w, 3)"
            )

        h, w, _ = image.shape
        if h > w:
            h_new = 512
            w_new = int(w * (512 / h))
        else:
            w_new = 512
            h_new = int(h * (512 / w))

        scaled = tf.image.resize(
            tf.expand_dims(image, axis=0),
            [h_new, w_new],
            method=tf.image.ResizeMethod.BICUBIC
        )
        scaled = scaled / 255.0
        scaled = tf.clip_by_value(scaled, 0.0, 1.0)

        return scaled

    def load_model(self):
        """
        Creates VGG19 model with
        AveragePooling layers.
        """
        vgg = tf.keras.applications.VGG19(
            include_top=False,
            weights='imagenet'
        )
        vgg.trainable = False

        outputs_dict = {}

        x = vgg.input
        for layer in vgg.layers[1:]:
            if isinstance(
                layer,
                tf.keras.layers.MaxPooling2D
            ):
                layer = tf.keras.layers.AveragePooling2D(
                    name=layer.name
                )
            layer.trainable = False
            x = layer(x)
            if (layer.name in self.style_layers or
                    layer.name == self.content_layer):
                outputs_dict[layer.name] = x

        style_outputs = [
            outputs_dict[layer]
            for layer in self.style_layers
        ]
        content_output = outputs_dict[self.content_layer]

        model_outputs = style_outputs + [content_output]

        self.model = tf.keras.Model(
            inputs=vgg.input,
            outputs=model_outputs
        )

    @staticmethod
    def gram_matrix(input_layer):
        """
        Calculates the gram matrix
        of an input layer tensor.
        """
        if (not isinstance(input_layer, (tf.Tensor, tf.Variable)) or
                len(input_layer.shape) != 4):
            raise TypeError(
                "input_layer must be a "
                "tensor of rank 4"
            )

        image_shape = tf.shape(input_layer)
        h = tf.cast(image_shape[1], tf.float32)
        w = tf.cast(image_shape[2], tf.float32)

        gram = tf.linalg.einsum(
            'bijc,bijd->bcd',
            input_layer,
            input_layer
        )
        return gram / (h * w)

    def generate_features(self):
        """
        Extracts features used to
        calculate neural style cost.
        """
        style_prep = tf.keras.applications.vgg19.preprocess_input(
            self.style_image * 255.0
        )
        content_prep = tf.keras.applications.vgg19.preprocess_input(
            self.content_image * 255.0
        )

        style_outputs = self.model(style_prep)
        content_outputs = self.model(content_prep)

        self.gram_style_features = [
            self.gram_matrix(layer)
            for layer in style_outputs[:len(self.style_layers)]
        ]
        self.content_feature = content_outputs[-1]

    def layer_style_cost(self, style_output, gram_target):
        """
        Calculates style cost for a single layer.
        """
        if (not isinstance(style_output, (tf.Tensor, tf.Variable)) or
                len(style_output.shape) != 4):
            raise TypeError(
                "style_output must be a "
                "tensor of rank 4"
            )

        c = style_output.shape[-1]
        if (not isinstance(gram_target, (tf.Tensor, tf.Variable)) or
                gram_target.shape != (1, c, c)):
            raise TypeError(
                f"gram_target must be a tensor "
                f"of shape [1, {c}, {c}]"
            )

        gram_style = self.gram_matrix(style_output)
        return tf.reduce_mean(
            tf.square(gram_style - gram_target)
        )

    def style_cost(self, style_outputs):
        """
        Calculates style cost across
        all style layers.
        """
        len_style = len(self.style_layers)
        if (not isinstance(style_outputs, list) or
                len(style_outputs) != len_style):
            raise TypeError(
                f"style_outputs must be a list "
                f"with a length of {len_style}"
            )

        weight = 1.0 / len_style
        style_cost_val = 0.0

        for i in range(len_style):
            layer_cost = self.layer_style_cost(
                style_outputs[i],
                self.gram_style_features[i]
            )
            style_cost_val += weight * layer_cost

        return style_cost_val

    def content_cost(self, content_output):
        """
        Calculates content cost for generated image.
        """
        s = self.content_feature.shape
        if (not isinstance(content_output, (tf.Tensor, tf.Variable)) or
                content_output.shape != s):
            raise TypeError(
                f"content_output must be a "
                f"tensor of shape {s}"
            )

        return 0.5 * tf.reduce_mean(
            tf.square(content_output - self.content_feature)
        )

    @staticmethod
    def variational_cost(generated_image):
        """
        Calculates variational cost
        for the generated image.
        """
        if (not isinstance(generated_image, (tf.Tensor, tf.Variable)) or
                len(generated_image.shape) != 4):
            raise TypeError(
                "generated_image must be a "
                "tensor of rank 4"
            )

        h = generated_image.shape[1]
        w = generated_image.shape[2]

        h_cost = tf.reduce_sum(
            tf.abs(
                generated_image[:, :-1, :, :] -
                generated_image[:, 1:, :, :]
            )
        )
        w_cost = tf.reduce_sum(
            tf.abs(
                generated_image[:, :, :-1, :] -
                generated_image[:, :, 1:, :]
            )
        )

        return (h_cost + w_cost) / tf.cast(h * w, tf.float32)

    def total_cost(self, generated_image):
        """
        Calculates the total cost
        for the generated image.
        """
        s = self.content_image.shape
        if (not isinstance(generated_image, (tf.Tensor, tf.Variable)) or
                generated_image.shape != s):
            raise TypeError(
                f"generated_image must be a "
                f"tensor of shape {s}"
            )

        preprocessed = tf.keras.applications.vgg19.preprocess_input(
            generated_image * 255.0
        )

        outputs = self.model(preprocessed)
        style_outputs = outputs[:len(self.style_layers)]
        content_output = outputs[-1]

        J_content = self.content_cost(content_output)
        J_style = self.style_cost(style_outputs)
        J_var = self.variational_cost(generated_image)

        J = (self.alpha * J_content +
             self.beta * J_style +
             self.var * J_var)

        return J, J_content, J_style, J_var

    def generate_image(
        self, iterations=1000, step=None,
        lr=0.01, beta1=0.9, beta2=0.99
    ):
        """
        Generates stylized image
        using gradient descent.
        """
        if not isinstance(iterations, int) or isinstance(iterations, bool):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be positive")

        if step is not None:
            if not isinstance(step, int) or isinstance(step, bool):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError(
                    "step must be positive "
                    "and less than iterations"
                )

        if not isinstance(lr, (int, float)) or isinstance(lr, bool):
            raise TypeError("lr must be a number")
        if lr <= 0:
            raise ValueError("lr must be positive")

        if not isinstance(beta1, float):
            raise TypeError("beta1 must be a float")
        if not (0 <= beta1 <= 1):
            raise ValueError(
                "beta1 must be in the range [0, 1]"
            )

        if not isinstance(beta2, float):
            raise TypeError("beta2 must be a float")
        if not (0 <= beta2 <= 1):
            raise ValueError(
                "beta2 must be in the range [0, 1]"
            )

        generated_image = tf.Variable(
            self.content_image,
            dtype=tf.float32,
            trainable=True
        )

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr,
            beta_1=beta1,
            beta_2=beta2
        )

        best_cost = float('inf')
        best_image = None

        for i in range(iterations + 1):
            with tf.GradientTape() as tape:
                J_total, J_content, J_style, J_var = self.total_cost(
                    generated_image
                )

            grads = tape.gradient(
                J_total,
                generated_image
            )
            optimizer.apply_gradients(
                [(grads, generated_image)]
            )
            generated_image.assign(
                tf.clip_by_value(
                    generated_image,
                    0.0, 1.0
                )
            )

            current_cost = float(J_total.numpy())
            if current_cost < best_cost:
                best_cost = current_cost
                best_image = tf.identity(generated_image)

            if step is not None and (
                i % step == 0 or i == iterations
            ):
                print(
                    f"Cost at iteration {i}: {current_cost}, "
                    f"content {float(J_content.numpy())}, "
                    f"style {float(J_style.numpy())}, "
                    f"var {float(J_var.numpy())}"
                )

        return best_image[0], best_cost
