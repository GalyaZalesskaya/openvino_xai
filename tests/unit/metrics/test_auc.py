import numpy as np

from openvino_xai.metrics.insertion_deletion_auc import InsertionDeletionAUC

import json

import cv2
import openvino as ov
import pytest
from typing import Callable, List, Mapping

from openvino_xai import Task
from openvino_xai.common.utils import retrieve_otx_model
from openvino_xai.explainer.explainer import Explainer, ExplainMode
from openvino_xai.explainer.utils import get_postprocess_fn, get_preprocess_fn, sigmoid
from openvino_xai.methods.black_box.base import Preset
from openvino_xai.metrics.pointing_game import PointingGame
from tests.unit.explanation.test_explanation_utils import VOC_NAMES

MODEL_NAME = "mlc_mobilenetv3_large_voc"


def postprocess_fn(x: Mapping):
    # Implementing own post-process function based on model's implementation
    # Return "logits" model output
    x = sigmoid(x)
    return x[0]

class TestAUC():

    image = cv2.imread("tests/assets/cheetah_person.jpg")
    preprocess_fn = get_preprocess_fn(
        change_channel_order=True,
        input_size=(224, 224),
        hwc_to_chw=True,
    )
    steps = 10

    @pytest.fixture(autouse=True)
    def setup(self, fxt_data_root):
        self.data_dir = fxt_data_root
        retrieve_otx_model(self.data_dir, MODEL_NAME)
        model_path = self.data_dir / "otx_models" / (MODEL_NAME + ".xml")
        core = ov.Core()
        model = core.read_model(model_path)
        compiled_model = core.compile_model(model=model, device_name="AUTO")
        self.auc = InsertionDeletionAUC(compiled_model, self.preprocess_fn, postprocess_fn)

        self.explainer = Explainer(
            model=model,
            task=Task.CLASSIFICATION,
            preprocess_fn=self.preprocess_fn,
            explain_mode=ExplainMode.WHITEBOX,
        )

    def test_auc_random_image(self):
        input_image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        saliency_map = np.random.rand(224, 224)

        insertion_auc_score = self.auc.insertion_auc_image(input_image, saliency_map, self.steps)
        assert insertion_auc_score >= 0.2

        deletion_auc_score = self.auc.deletion_auc_image(input_image, saliency_map, self.steps)
        assert deletion_auc_score >= 0.2

    # def test_auc_image(self):
    #     explanation = self.explainer(
    #         self.image,
    #         targets=["person"],
    #         label_names=VOC_NAMES,
    #         preset=Preset.SPEED,
    #         colormap=False,
    #     )
    #     saliency_map = list(explanation.saliency_map.values())[0]
    #     insertion_auc_score = self.auc.insertion_auc_image(self.image, saliency_map, self.steps)
    #     assert insertion_auc_score >= 0.9

    #     deletion_auc_score = self.auc.deletion_auc_image(self.image, saliency_map, self.steps)
    #     assert deletion_auc_score >= 0.2

    # def test_auc_evaluate(self):
    #     # TODO support multiple classes
    #     images = [self.image, self.image]
    #     saliency_maps = []
    #     for image in images:
    #         explanation = self.explainer(
    #             image,
    #             targets=["person"],
    #             label_names=VOC_NAMES,
    #             preset=Preset.SPEED,
    #             colormap=False,
    #         )
    #         saliency_map = list(explanation.saliency_map.values())[0]
    #         saliency_maps.append(saliency_map) 
    #     insertion, deletion, delta = self.auc.evaluate(images, saliency_maps, self.steps)
    #     assert insertion >= 0.9
    #     assert deletion >= 0.2
    #     assert delta >= 0.7

    

