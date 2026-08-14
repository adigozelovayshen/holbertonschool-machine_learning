#!/usr/bin/env python3
"""
YOLO v3 Object Detection Initialization Module
"""
import tensorflow.keras as K


class Yolo:
    """
    Class Yolo that uses the YOLO v3 algorithm to perform object detection.
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Class constructor for Yolo.

        Parameters:
            model_path (str): Path to where a Darknet Keras model is stored.
            classes_path (str): Path to where the list of class names is stored.
            class_t (float): Box score threshold for initial filtering step.
            nms_t (float): IOU threshold for non-max suppression.
            anchors (numpy.ndarray): Array of shape (outputs, anchor_boxes, 2)
                                     containing all anchor box dimensions.
        """
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
