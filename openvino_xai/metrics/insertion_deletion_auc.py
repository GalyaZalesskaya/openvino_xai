from typing import List, Tuple

import numpy as np
# from sklearn.metrics import auc
import cv2

def auc(arr):
    """Returns normalized Area Under Curve of the array."""
    return (arr.sum() - arr[0] / 2 - arr[-1] / 2) / (arr.shape[0] - 1)


class InsertionDeletionAUC:

    def __init__(self, compiled_model, preprocess_fn, postprocess_fn):
        self.preprocess_fn = preprocess_fn
        self.postprocess_fn = postprocess_fn
        self.compiled_model = compiled_model

    def predict(self, input) -> int:
        logits = self.compiled_model([self.preprocess_fn(input)])[0]
        logits = self.postprocess_fn(logits)
        predicted_class = np.argmax(logits)
        return logits, predicted_class


    def insertion_auc_image(self, input_image, saliency_map, steps=100, baseline_value=0):
        """
        Calculate the Insertion AUC metric for images.

        Parameters:
        - model: the model to evaluate.
        - input_image: the input image to the model (H, W, C).
        - saliency_map: importance scores for each pixel (H, W).
        - steps: number of steps for inserting pixels.
        - baseline_value: value to initialize the baseline (e.g., 0 for a black image).

        Returns:
        - insertion_auc_score: the calculated AUC for insertion.
        """
        baseline = np.full_like(input_image, 0)
        sorted_indices = np.argsort(-saliency_map.flatten())  # Sort pixels by importance (descending)
        sorted_indices = np.unravel_index(sorted_indices, saliency_map.shape)

        _, pred_class = self.predict(input_image)

        scores = []
        for i in range(steps + 1):
            temp_image = baseline.copy()
            num_pixels_to_insert = int(i * len(sorted_indices[0]) / steps)
            temp_image[sorted_indices[0][:num_pixels_to_insert], sorted_indices[1][:num_pixels_to_insert]] = input_image[sorted_indices[0][:num_pixels_to_insert], sorted_indices[1][:num_pixels_to_insert]]

            # Predict and record the score
            # cv2.imshow("temp_image", temp_image)
            temp_logits, _  = self.predict(temp_image)  # Model expects batch dimension
            scores.append(temp_logits[pred_class])
        #     cv2.waitKey(0)
        # cv2.destroyAllWindows()

        insertion_auc_score = auc(np.array(scores))
        return insertion_auc_score


    def deletion_auc_image(self, input_image, saliency_map, steps=100):
        """
        Calculate the Deletion AUC metric for images.

        Parameters:
        - model: the model to evaluate.
        - input_image: the input image to the model (H, W, C).
        - saliency_map: importance scores for each pixel (H, W).
        - steps: number of steps for deleting pixels.

        Returns:
        - deletion_auc_score: the calculated AUC for deletion.
        """
        sorted_indices = np.argsort(-saliency_map.flatten())  # Sort pixels by importance (descending)
        sorted_indices = np.unravel_index(sorted_indices, saliency_map.shape)

        _, pred_class = self.predict(input_image)

        scores = []
        for i in range(steps + 1):
            temp_image = input_image.copy()
            num_pixels_to_delete = int(i * len(sorted_indices[0]) / steps)
            temp_image[sorted_indices[0][:num_pixels_to_delete], sorted_indices[1][:num_pixels_to_delete]] = 0  # Remove important pixels

            # Predict and record the score
            # cv2.imshow("temp_image", temp_image)
            temp_logits, _  = self.predict(temp_image)  # Model expects batch dimension
            scores.append(temp_logits[pred_class])
        #     cv2.waitKey(0)
        # cv2.destroyAllWindows()

        deletion_auc_score = auc(np.array(scores))
        return deletion_auc_score

    def evaluate(self, input_images, saliency_maps, steps):
        insertions, deletions = [], []
        for input_image, saliency_map in zip(input_images, saliency_maps):
            insertion = self.insertion_auc_image(input_image, saliency_map,steps)
            deletion = self.deletion_auc_image(input_image, saliency_map, steps)

            insertions.append(insertion)
            deletions.append(deletion)
        
        insertion = np.mean(np.array(insertions))
        deletion = np.mean(np.array(deletion))
        delta = insertion - deletion

        return insertion, deletion, delta
