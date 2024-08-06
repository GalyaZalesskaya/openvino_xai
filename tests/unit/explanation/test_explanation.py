# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

import numpy as np
import pytest

from openvino_xai.explainer.explanation import Explanation
from tests.unit.explanation.test_explanation_utils import VOC_NAMES

SALIENCY_MAPS = (np.random.rand(1, 20, 5, 5) * 255).astype(np.uint8)
SALIENCY_MAPS_IMAGE = (np.random.rand(1, 5, 5) * 255).astype(np.uint8)


class TestExplanation:
    def test_targets(self):
        explain_targets = [0, 2]
        explanation_indices = Explanation(
            SALIENCY_MAPS,
            targets=explain_targets,
            label_names=VOC_NAMES,
        )

        explain_targets = ["aeroplane", "bird"]
        explanation_names = Explanation(
            SALIENCY_MAPS,
            targets=explain_targets,
            label_names=VOC_NAMES,
        )

        sm_indices = explanation_indices.saliency_map
        sm_names = explanation_names.saliency_map
        assert len(sm_indices) == len(sm_names)
        assert set(sm_indices.keys()) == set(sm_names.keys()) == {0, 2}

    def test_shape(self):
        explanation = self._get_explanation()
        assert explanation.shape == (5, 5)

    def test_save(self, tmp_path):
        save_path = tmp_path / "saliency_maps"

        explanation = self._get_explanation()
        explanation.save(save_path, "test_map")
        assert os.path.isfile(save_path / "test_map_target_aeroplane.jpg")
        assert os.path.isfile(save_path / "test_map_target_bird.jpg")

        explanation = self._get_explanation()
        explanation.save(save_path)
        assert os.path.isfile(save_path / "target_aeroplane.jpg")
        assert os.path.isfile(save_path / "target_bird.jpg")

        explanation = self._get_explanation(label_names=None)
        explanation.save(save_path, "test_map")
        assert os.path.isfile(save_path / "test_map_target_0.jpg")
        assert os.path.isfile(save_path / "test_map_target_2.jpg")

        explanation = self._get_explanation(saliency_maps=SALIENCY_MAPS_IMAGE, label_names=None)
        explanation.save(save_path, "test_map")
        assert os.path.isfile(save_path / "test_map.jpg")

    def _get_explanation(self, saliency_maps=SALIENCY_MAPS, label_names=VOC_NAMES):
        explain_targets = [0, 2]
        explanation = Explanation(
            saliency_maps,
            targets=explain_targets,
            label_names=label_names,
        )
        return explanation

    def test_plot(self, capsys):
        explanation = self._get_explanation()

        # Matplotloib backend
        explanation.plot([0, 2], backend="matplotlib")
        # Targets as label names
        explanation.plot(["aeroplane", "bird"], backend="matplotlib")

        # CV backend
        explanation._plot_cv([0, 2], wait_time=1)

        # Invalid backend
        with pytest.raises(ValueError):
            explanation.plot([0, 1], backend="invalid")

        # Class index that is not in saliency maps will be ommitted with warning
        explanation.plot([0, 3])
        captured = capsys.readouterr()
        assert "Provided class index 3 is not available among saliency maps." in captured.out
