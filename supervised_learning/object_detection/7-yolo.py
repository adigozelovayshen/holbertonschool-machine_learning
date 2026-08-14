#!/usr/bin/env python3
"""
YOLO v3 Object Detection Module with Full Prediction Pipeline
"""
import cv2
import glob
import numpy as np
import os
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

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Filters bounding boxes based on class score threshold.

        Parameters:
            boxes (list): List of numpy.ndarrays of shape
                (grid_height, grid_width, anchor_boxes, 4)
            box_confidences (list): List of numpy.ndarrays of shape
                (grid_height, grid_width, anchor_boxes, 1)
            box_class_probs (list): List of numpy.ndarrays of shape
                (grid_height, grid_width, anchor_boxes, classes)

        Returns:
            tuple: (filtered_boxes, box_classes, box_scores)
        """
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for i in range(len(boxes)):
            scores = box_confidences[i] * box_class_probs[i]
            classes = np.argmax(scores, axis=-1)
            class_scores = np.max(scores, axis=-1)

            mask = class_scores >= self.class_t

            filtered_boxes.append(boxes[i][mask])
            box_classes.append(classes[mask])
            box_scores.append(class_scores[mask])

        filtered_boxes = np.concatenate(filtered_boxes, axis=0)
        box_classes = np.concatenate(box_classes, axis=0)
        box_scores = np.concatenate(box_scores, axis=0)

        return filtered_boxes, box_classes, box_scores

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Applies Non-Max Suppression (NMS) on filtered bounding boxes.

        Parameters:
            filtered_boxes (numpy.ndarray): Shape (?, 4) filtered boxes.
            box_classes (numpy.ndarray): Shape (?,) class predictions.
            box_scores (numpy.ndarray): Shape (?) box scores.

        Returns:
            tuple: (box_predictions, predicted_box_classes,
                    predicted_box_scores)
        """
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        for c in np.unique(box_classes):
            idx = np.where(box_classes == c)[0]
            cls_boxes = filtered_boxes[idx]
            cls_scores = box_scores[idx]

            sort_idx = np.argsort(cls_scores)[::-1]
            cls_boxes = cls_boxes[sort_idx]
            cls_scores = cls_scores[sort_idx]

            while len(cls_scores) > 0:
                box_predictions.append(cls_boxes[0])
                predicted_box_classes.append(c)
                predicted_box_scores.append(cls_scores[0])

                if len(cls_scores) == 1:
                    break

                box = cls_boxes[0]
                rest_boxes = cls_boxes[1:]

                x1 = np.maximum(box[0], rest_boxes[:, 0])
                y1 = np.maximum(box[1], rest_boxes[:, 1])
                x2 = np.minimum(box[2], rest_boxes[:, 2])
                y2 = np.minimum(box[3], rest_boxes[:, 3])

                inter_w = np.maximum(0, x2 - x1)
                inter_h = np.maximum(0, y2 - y1)
                inter_area = inter_w * inter_h

                box_area = (box[2] - box[0]) * (box[3] - box[1])
                rest_area = (
                    (rest_boxes[:, 2] - rest_boxes[:, 0]) *
                    (rest_boxes[:, 3] - rest_boxes[:, 1])
                )

                union_area = box_area + rest_area - inter_area
                iou = inter_area / union_area

                keep_idx = np.where(iou <= self.nms_t)[0]
                cls_boxes = rest_boxes[keep_idx]
                cls_scores = cls_scores[1:][keep_idx]

        box_predictions = np.array(box_predictions)
        predicted_box_classes = np.array(predicted_box_classes)
        predicted_box_scores = np.array(predicted_box_scores)

        return box_predictions, predicted_box_classes, predicted_box_scores

    @staticmethod
    def load_images(folder_path):
        """
        Loads all images from a specified folder path.

        Parameters:
            folder_path (str): Path to folder holding images to load.

        Returns:
            tuple: (images, image_paths)
                - images: list of numpy.ndarrays representing loaded images.
                - image_paths: list of paths to individual images in images.
        """
        image_paths = glob.glob(folder_path + '/*', recursive=False)
        images = [cv2.imread(image_path) for image_path in image_paths]
        return images, image_paths

    def preprocess_images(self, images):
        """
        Preprocesses images for the Darknet model.

        Parameters:
            images (list): List of images as numpy.ndarrays.

        Returns:
            tuple: (pimages, image_shapes)
                - pimages: numpy.ndarray of shape (ni, input_h, input_w, 3)
                  containing all preprocessed images.
                - image_shapes: numpy.ndarray of shape (ni, 2) containing
                  original height and width of each image.
        """
        try:
            input_w = self.model.input.shape[1]
            input_h = self.model.input.shape[2]
        except AttributeError:
            input_w = self.model.input[0].shape[1]
            input_h = self.model.input[0].shape[2]

        pimages = []
        image_shapes = []

        for img in images:
            image_shapes.append(img.shape[:2])
            resized = cv2.resize(
                img, (input_w, input_h),
                interpolation=cv2.INTER_CUBIC
            )
            rescaled = resized / 255.0
            pimages.append(rescaled)

        pimages = np.array(pimages)
        image_shapes = np.array(image_shapes)

        return pimages, image_shapes

    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        """
        Displays the image with all boundary boxes, class names, and scores.

        Parameters:
            image (numpy.ndarray): Unprocessed image.
            boxes (numpy.ndarray): Boundary boxes for the image.
            box_classes (numpy.ndarray): Class indices for each box.
            box_scores (numpy.ndarray): Box scores for each box.
            file_name (str): File path where original image is stored.
        """
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.astype(int)
            class_name = self.class_names[box_classes[i]]
            score = box_scores[i]
            label = f"{class_name} {score:.2f}"

            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(
                image,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA
            )

        cv2.imshow(file_name, image)
        key = cv2.waitKey(0) & 0xFF

        if key == ord('s'):
            if not os.path.exists('detections'):
                os.makedirs('detections')
            cv2.imwrite(os.path.join('detections', file_name), image)

        cv2.destroyAllWindows()

    def predict(self, folder_path):
        """
        Predicts objects in all images within a specified folder.

        Parameters:
            folder_path (str): Path to folder holding images to predict.

        Returns:
            tuple: (predictions, image_paths)
                - predictions: list of tuples for each image containing
                  (boxes, box_classes, box_scores)
                - image_paths: list of image paths corresponding to predictions
        """
        images, image_paths = self.load_images(folder_path)
        pimages, image_shapes = self.preprocess_images(images)

        outputs = self.model.predict(pimages)

        predictions = []

        for i in range(len(images)):
            image_outputs = [output[i] for output in outputs]
            boxes, confs, probs = self.process_outputs(
                image_outputs, image_shapes[i]
            )
            f_boxes, f_classes, f_scores = self.filter_boxes(
                boxes, confs, probs
            )
            p_boxes, p_classes, p_scores = self.non_max_suppression(
                f_boxes, f_classes, f_scores
            )

            predictions.append((p_boxes, p_classes, p_scores))

            file_name = os.path.basename(image_paths[i])
            self.show_boxes(
                images[i], p_boxes, p_classes, p_scores, file_name
            )

        return predictions, image_paths
