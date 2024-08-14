import unittest
from typing import List, Tuple

import numpy as np

from openvino_xai.metrics.pointing_game import PointingGame


class TestPointingGame(unittest.TestCase):
    def setUp(self):
        self.pointing_game = PointingGame()

    def test_pointing_game(self):
        saliency_map = np.zeros((3, 3), dtype=np.float32)
        saliency_map[1, 1] = 1

        ground_truth_bbox = (1, 1, 1, 1)
        score = self.pointing_game.pointing_game(saliency_map, ground_truth_bbox)
        assert score == 1

        ground_truth_bbox = (0, 0, 0, 0)
        score = self.pointing_game.pointing_game(saliency_map, ground_truth_bbox)
        assert score == 0

    def test_pointing_game_evaluate(self):
        saliency_map = np.zeros((3, 3), dtype=np.float32)
        saliency_map[1, 1] = 1

        saliency_maps = [saliency_map, saliency_map]
        ground_truth_bboxes = [(0, 0, 0, 0), (1, 1, 1, 1)]
        score = self.pointing_game.evaluate(saliency_maps, ground_truth_bboxes)
        assert score == 0.5
