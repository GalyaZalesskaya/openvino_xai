from abc import ABC
from abc import abstractmethod

import numpy as np

from openvino_xai.model import XAIModel, XAIClassificationModel
from openvino_xai.utils import logger


class PostProcessor:
    """Processor implements post-processing for the saliency map."""

    @staticmethod
    def postprocess(saliency_map, data=None, normalize=True, resize=False, colormap=False, overlay=False):
        """Interface for postprocess method."""
        if normalize:
            saliency_map = PostProcessor._normalize(saliency_map)
        if resize:
            if data is None:
                # TODO: add explicit target_size as an option
                raise ValueError("Input data has to be provided for resize (for target size estimation).")
            saliency_map = PostProcessor._resize(saliency_map, data)
        if colormap:
            saliency_map = PostProcessor._colormap(saliency_map)
        if overlay:
            if data is None:
                raise ValueError("Input data has to be provided for overlay.")
            if not PostProcessor._resized(data, saliency_map):
                saliency_map = PostProcessor._resize(saliency_map, data)
            if not PostProcessor._colormapped(saliency_map):
                saliency_map = PostProcessor._colormap(saliency_map)
            saliency_map = PostProcessor._overlay(saliency_map, data)
        return saliency_map

    @staticmethod
    def _normalize(saliency_map):
        # TODO: move norm here from IR
        return saliency_map

    @staticmethod
    def _resize(saliency_map, data):
        return saliency_map

    @staticmethod
    def _colormap(saliency_map):
        return saliency_map

    @staticmethod
    def _overlay(saliency_map, data):
        return saliency_map

    @staticmethod
    def _resized(saliency_map, data):
        return True

    @staticmethod
    def _colormapped(saliency_map):
        return True


class Explainer(ABC):
    """A base interface for explainer."""

    def __init__(self, model):
        self._model = model
        self._processor = PostProcessor()

    @abstractmethod
    def explain(self, data):
        """Explain the input."""
        raise NotImplementedError

    @staticmethod
    def _check_data_type(saliency_map):
        if saliency_map.dtype != np.uint8:
            saliency_map = saliency_map.astype(np.uint8)
        return saliency_map


class WhiteBoxExplainer(Explainer):
    """Explainer explains models with XAI branch injected."""

    def explain(self, data, explain_only_predictions=False):
        """Explain the input in white box mode."""
        raw_result = self._model(data)

        saliency_map = raw_result.saliency_map
        if saliency_map.size == 0:
            raise RuntimeError("Model does not contain saliency_map output.")

        saliency_map = self._check_data_type(saliency_map)
        # TODO: if explain_only_predictions: keep saliency maps only for predicted classes
        saliency_map = self._processor.postprocess(saliency_map, data=data)
        return saliency_map


class BlackBoxExplainer(Explainer):
    """Base class for explainers that consider model as a black-box."""


class RISEExplainer(BlackBoxExplainer):
    def explain(self, data):
        """Explain the input."""
        raise NotImplementedError


class DRISEExplainer(BlackBoxExplainer):
    def explain(self, data):
        """Explain the input."""
        raise NotImplementedError


class AutoExplainer(Explainer):
    """Explain in auto mode, using white box or black box approach."""

    def __init__(self, model, explain_parameters=None):
        super().__init__(model)
        self._explain_parameters = explain_parameters if explain_parameters else {}


class ClassificationAutoExplainer(AutoExplainer):
    """Explain classification models in auto mode, using white box or black box approach."""

    def explain(self, data):
        """
        Implements three explain scenarios, for different IR models:
            1. IR model contain xai branch -> infer Model API wrapper.
            2. If not (1), IR model can be augmented with XAI branch -> augment and infer.
            3. If not (1) and (2), IR model can NOT be augmented with XAI branch -> use XAI BB method.

        Args:
            data(numpy.ndarray): data to explain.
        """
        if XAIModel.has_xai(self._model):
            logger.info("Model already has XAI - using White Box explainer.")
            explanations = WhiteBoxExplainer(self._model).explain(data)
            return explanations
        else:
            try:
                logger.info("Model does not have XAI - trying to insert XAI and use White Box explainer.")
                self._model = XAIClassificationModel.insert_xai(self._model, self._explain_parameters)
                explanations = WhiteBoxExplainer(self._model).explain(data)
                return explanations
            except Exception as e:
                print(e)
                logger.info("Failed to insert XAI into the model. Calling Black Box explainer.")
                explanations = RISEExplainer(self._model).explain(data)
                return explanations


class DetectionAutoExplainer(AutoExplainer):
    """Explain detection models in auto mode, using white box or black box approach."""

    def explain(self, data):
        """Explain the input."""
        raise NotImplementedError