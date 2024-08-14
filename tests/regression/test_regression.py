# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json

import cv2
import openvino as ov
import pytest

from openvino_xai import Task
from openvino_xai.common.utils import retrieve_otx_model
from openvino_xai.explainer.explainer import Explainer, ExplainMode
from openvino_xai.explainer.utils import get_postprocess_fn, get_preprocess_fn
from openvino_xai.methods.black_box.base import Preset
from openvino_xai.metrics.pointing_game import PointingGame
from tests.unit.explanation.test_explanation_utils import VOC_NAMES

MODEL_NAME = "mlc_mobilenetv3_large_voc"


def load_gt_bboxes(class_name="person"):
    with open("tests/assets/cheetah_person_coco.json", "r") as f:
        coco_anns = json.load(f)

    category_id = [category["id"] for category in coco_anns["categories"] if category["name"] == class_name]
    category_id = category_id[0]

    category_gt_bboxes = [
        annotation["bbox"] for annotation in coco_anns["annotations"] if annotation["category_id"] == category_id
    ]
    return category_gt_bboxes


class TestDummyRegression:
    image = cv2.imread("tests/assets/cheetah_person.jpg")

    preprocess_fn = get_preprocess_fn(
        change_channel_order=True,
        input_size=(224, 224),
        hwc_to_chw=True,
    )
    gt_bboxes = load_gt_bboxes()
    pointing_game = PointingGame()

    @pytest.fixture(autouse=True)
    def setup(self, fxt_data_root):
        self.data_dir = fxt_data_root

    def test_explainer(self):
        retrieve_otx_model(self.data_dir, MODEL_NAME)
        model_path = self.data_dir / "otx_models" / (MODEL_NAME + ".xml")
        model = ov.Core().read_model(model_path)

        explain_mode = ExplainMode.WHITEBOX
        explainer = Explainer(
            model=model,
            task=Task.CLASSIFICATION,
            preprocess_fn=self.preprocess_fn,
            postprocess_fn=get_postprocess_fn(),
            explain_mode=explain_mode,
        )

        explanation = explainer(
            self.image,
            targets=["person"],
            label_names=VOC_NAMES,
            preset=Preset.SPEED,
            colormap=False,
        )
        assert len(explanation.saliency_map) == 1

        # For now, assume that there's only one class
        # TODO: support multiple classes
        saliency_maps = list(explanation.saliency_map.values())
        score = self.pointing_game.evaluate(saliency_maps, self.gt_bboxes)
        assert score > 0.5
