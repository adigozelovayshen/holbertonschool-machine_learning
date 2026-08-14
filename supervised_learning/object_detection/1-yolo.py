#!/usr/bin/env python3
"""
YOLO v3 Object Detection Module with Output Processing
"""
import numpy as np
import tensorflow.keras as K


class Yolo:
    """
    Class Yolo that uses the YOLO v3 algorithm to perform object detection.
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Class constructor for Yolo.

        Parameters:
            model_path (str): Path to Darknet Keras model.
            classes_path (str): Path to file with class names.
            class_t (float): Box score threshold for initial filtering step.
            nms_t (float): IOU threshold for non-max suppression.
            anchors (numpy.ndarray): Array of anchor box dimensions.
        """
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """
        Processes Darknet model predictions for a single image.

        Parameters:
            outputs (list): List of numpy.ndarrays containing predictions.
            image_size (numpy.ndarray): Image's original size [height, width].

        Returns:
            tuple: (boxes, box_confidences, box_class_probs)
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        image_height, image_width = image_size[0], image_size[1]

        try:
            input_width = self.model.input.shape[1]
            input_height = self.model.input.shape[2]
        except AttributeError:
            input_width = self.model.input[0].shape[1]
            input_height = self.model.input[0].shape[2]

        for i, output in enumerate(outputs):
            grid_height, grid_width, anchor_boxes, _ = output.shape

            t_x = output[..., 0]
            t_y = output[..., 1]
            t_w = output[..., 2]
            t_h = output[..., 3]

            conf = 1 / (1 + np.exp(-output[..., 4:5]))
            probs = 1 / (1 + np.exp(-output[..., 5:]))

            box_confidences.append(conf)
            box_class_probs.append(probs)

            c_x = np.tile(np.arange(grid_width), (grid_height, 1))
            c_x = np.tile(c_x[..., np.newaxis], (1, 1, anchor_boxes))

            c_y = np.tile(
                np.arange(grid_height)[:, np.newaxis], (1, grid_width)
            )
            c_y = np.tile(c_y[..., np.newaxis], (1, 1, anchor_boxes))

            b_x = (1 / (1 + np.exp(-t_x)) + c_x) / grid_width
            b_y = (1 / (1 + np.exp(-t_y)) + c_y) / grid_height

            anchor_w = self.anchors[i, :, 0]
            anchor_h = self.anchors[i, :, 1]

            b_w = (anchor_w * np.exp(t_w)) / input_width
            b_h = (anchor_h * np.exp(t_h)) / input_height

            center_x = b_x * image_width
            center_y = b_y * image_height
            w = b_w * image_width
            h = b_h * image_height

            x1 = center_x - (w / 2)
            y1 = center_y - (h / 2)
            x2 = center_x + (w / 2)
            y2 = center_y + (h / 2)

            box = np.zeros(output[..., :4].shape)
            box[..., 0] = x1
            box[..., 1] = y1
            box[..., 2] = x2
            box[..., 3] = y2

            boxes.append(box)

        return boxes, box_confidences, box_class_probs
