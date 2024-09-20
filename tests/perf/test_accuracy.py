# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
from enum import Enum
from typing import Any, Dict, List, Tuple
from time import time
from pathlib import Path
from tqdm import tqdm
import cv2
from openvino_xai.common.utils import scaling
from openvino_xai.common.utils import sigmoid, softmax
from openvino_xai.explainer.visualizer import colormap, overlay
import PIL

from openvino_xai.methods.black_box.base import Preset
from openvino_xai import insert_xai
from openvino_xai.common.utils import IdentityPreprocessFN, has_xai, logger
import numpy as np
import openvino as ov
import pytest
import pandas as pd
import torchvision.transforms as transforms

from openvino_xai import Task
from openvino_xai.common.utils import retrieve_otx_model
from openvino_xai.explainer.explainer import Explainer, ExplainMode
from openvino_xai.common.parameters import Method, Task, WhiteBoxXAIMethods, BlackBoxXAIMethods
from openvino_xai.explainer.utils import (
    ActivationType,
    get_postprocess_fn,
    get_preprocess_fn,
)
from openvino_xai.metrics import ADCC, InsertionDeletionAUC, PointingGame
from tests.unit.explainer.test_explanation_utils import VOC_NAMES, get_imagenet_labels
from tests.test_suite.dataset_utils import DatasetType, coco_anns_to_gt_bboxes, voc_anns_to_gt_bboxes, define_dataset_type

from openvino_xai.utils.model_export import export_to_ir, export_to_onnx

from tests.perf.custom_dataset import CustomVOCDetection

datasets = pytest.importorskip("torchvision.datasets")
timm = pytest.importorskip("timm")
torch = pytest.importorskip("torch")
import torchvision.models as models

IMAGENET_MODELS = [
    "resnet18.a1_in1k", 
    # "resnet50.a1_in1k",
    # "resnext50_32x4d.a1h_in1k",
    # "vgg16.tv_in1k"
    ]
VOC_MODELS = [
    # "mlc_mobilenetv3_large_voc"
    ]
TRANSFORMER_MODELS = [
    "deit_tiny_patch16_224.fb_in1k",  # Downloads last month 8,377
    # "deit_base_patch16_224.fb_in1k", # Downloads last month 6,323

    # "vit_tiny_patch16_224.augreg_in21k",  # Downloads last month 3,671 - trained on ImageNet-21k
    # "vit_base_patch16_224.augreg2_in21k_ft_in1k",  # Downloads last month 207,590 - trained on ImageNet-21k

]
TEST_MODELS = IMAGENET_MODELS + VOC_MODELS + TRANSFORMER_MODELS

IMAGENET_LABELS = get_imagenet_labels()

EXPLAIN_METHODS = [
    Method.RECIPROCAM, 
                   Method.AISE, 
                   Method.RISE, 
                   Method.ACTIVATIONMAP
                   ]

class TestAccuracy:

    def setup_model(self, data_dir, model_name):
        if model_name in VOC_MODELS:
            self.dataset_label_list = VOC_NAMES
            retrieve_otx_model(data_dir, model_name)
            model_path = data_dir / "otx_models" / (model_name + ".xml")
            model = ov.Core().read_model(model_path)
            return model, None

        elif model_name in IMAGENET_MODELS:
            self.dataset_label_list = IMAGENET_LABELS

            timm_model, model_cfg = self.get_timm_model(model_name)
            ir_path = data_dir / "timm_models" / "converted_models" / model_name / "model_fp32.xml"
            if not ir_path.is_file():
                output_model_dir = self.output_dir / "timm_models" / "converted_models" / model_name
                output_model_dir.mkdir(parents=True, exist_ok=True)
                ir_path = output_model_dir / "model_fp32.xml"
                input_size = [1] + list(model_cfg["input_size"])
                dummy_tensor = torch.rand(input_size)
                onnx_path = output_model_dir / "model_fp32.onnx"
                set_dynamic_batch = model_name in TRANSFORMER_MODELS
                export_to_onnx(timm_model, onnx_path, dummy_tensor, set_dynamic_batch)
                export_to_ir(onnx_path, output_model_dir / "model_fp32.xml")
            ov_model = ov.Core().read_model(ir_path)
            return ov_model, model_cfg
        else:
            raise ValueError(f"Model {model_name} is not supported since it's not VOC or ImageNet model.")
    
    def setup_process_fn(self, model_cfg):
        if self.model_name in VOC_MODELS:
            # VOC model
            self.preprocess_fn = get_preprocess_fn(
                change_channel_order=False,
                input_size=(224, 224),
                hwc_to_chw=True,
            )
            self.postprocess_fn = get_postprocess_fn(activation=ActivationType.SIGMOID)
        elif self.model_name in IMAGENET_MODELS+TRANSFORMER_MODELS:
            # ImageNet model
            mean_values = [(item * 255) for item in model_cfg["mean"]]
            scale_values = [(item * 255) for item in model_cfg["std"]]
            self.preprocess_fn = get_preprocess_fn(
                change_channel_order=True,
                input_size=model_cfg["input_size"][1:],
                mean=mean_values,
                std=scale_values,
                hwc_to_chw=True,
            )
            self.postprocess_fn = get_postprocess_fn(activation=ActivationType.SOFTMAX)
        else:
            raise ValueError(f"Model {self.model_name} is not supported since it's not VOC or ImageNet model.")

    def setup_explainer(self, model, explain_method):
        explain_mode = ExplainMode.WHITEBOX if explain_method in WhiteBoxXAIMethods else ExplainMode.BLACKBOX

        if self.model_name in TRANSFORMER_MODELS and explain_method == Method.RECIPROCAM:
            explain_method = Method.VITRECIPROCAM

        self.explainer = Explainer(
            model=model,
            task=Task.CLASSIFICATION,
            preprocess_fn=self.preprocess_fn,
            postprocess_fn=self.postprocess_fn,
            explain_mode=explain_mode,
            explain_method=explain_method,
            embed_scaling=True,
        )
        kwargs = {}
        if explain_method in BlackBoxXAIMethods:
            kwargs.update({"preset": Preset.SPEED})
        if explain_method == Method.RISE:
            kwargs.update({"num_masks": 5})
        return kwargs


    @pytest.fixture(autouse=True)
    def setup(self, fxt_output_root, fxt_data_root):
        self.data_dir = fxt_data_root
        # self.dataset_parameters = fxt_dataset_parameters

        self.output_dir = Path("/home/gzalessk/code/openvino_xai/tests/perf/validation_results")
        # self.output_dir = fxt_output_root

        # data_root, ann_path = self.dataset_parameters[0]

    @pytest.mark.parametrize("model_id", TEST_MODELS)
    @pytest.mark.parametrize("explain_method", EXPLAIN_METHODS)
    def test_explainer_images(self, model_id, fxt_dataset_parameters, explain_method):
        self.model_name = model_id

        self.setup_dataset(self.model_name, fxt_dataset_parameters)
        self.dataset_name  = "coco" if self.dataset_type == DatasetType.COCO else "voc"

        self.data_metric_path = self.output_dir / self.model_name / explain_method.value
        os.makedirs(self.data_metric_path, exist_ok=True)

        model, model_cfg = self.setup_model(self.data_dir, self.model_name)
        self.setup_process_fn(model_cfg)

        kwargs = self.setup_explainer(model, explain_method)

        self.pointing_game = PointingGame()
        self.auc = InsertionDeletionAUC(model, self.preprocess_fn, self.postprocess_fn)
        self.adcc = ADCC(model, self.preprocess_fn, self.postprocess_fn, self.explainer)

        records = []
        explained_images = 0
        experiment_start_time = time()
        max_num = 1#len(self.dataset)
        batch_size = 5000

        for lrange in range(0, max_num, batch_size):
            rrange = min(max_num, lrange+batch_size)

            start_time = time()
            images, explanations, dataset_gt_bboxes = [], [], []
            for i in range(lrange, rrange):
                image, anns = self.dataset[i]
                
                image_np = np.array(image) # PIL -> np.array
                gt_bbox_dict = self.anns_to_gt_bboxes(anns, self.dataset_labels_dict)

                # To measure the quality of predicted saliency maps without the gt info from ImageNet or COCO dataset (found out how to check it)
                # targets = np.argmax(self.model_predict(image_np))
                targets = list(gt_bbox_dict.keys())
                intersected_targets = list(set(targets) & set(self.dataset_label_list))
                if len(intersected_targets) == 0:
                    # Skip images where gt classes and model classes do not match
                    continue
                explanation = self.explainer(image_np, targets=intersected_targets, label_names=self.dataset_label_list, colormap=False, **kwargs)
                images.append(image_np)
                explanations.append(explanation)
                dataset_gt_bboxes.append(gt_bbox_dict)
            
            # Write per-batch statistics to track failures
            explained_images += len(explanations)
            record = {"range": f"{lrange}-{rrange}"}
            record.update(self.get_scores_times(explanations, images, dataset_gt_bboxes, start_time))
            records.append(record)

            df = pd.DataFrame([record]).round(3)
            df.to_csv(self.data_metric_path / f"accuracy_{self.dataset_name}.csv", mode='a', header=False, index=False)

        experiment_time = time() - experiment_start_time
        mean_scores_dict = {"explained_images": explained_images, "overall_time": experiment_time}
        mean_scores_dict.update({key: np.mean([record[key] for record in records if key in record]) for key in records[0].keys() if key!='range'})
        df = pd.DataFrame([mean_scores_dict]).round(3)
        df.to_csv(self.data_metric_path / f"mean_accuracy_{self.dataset_name}.csv", index=False)


    def get_scores_times(self, explanations, images, dataset_gt_bboxes, start_time):
        score = {}
        if len(explanations) == 0:
            return score

        explain_time = time() - start_time
        score["explain_time"] = explain_time

        previous_time = time()
        score.update(self.pointing_game.evaluate(explanations, dataset_gt_bboxes))
        score["pointing_game_time"] = time() - previous_time

        previous_time = time()
        score.update(self.auc.evaluate(explanations, images, steps=30))
        score["auc_time"] = time() - previous_time

        previous_time = time()
        score.update(self.adcc.evaluate(explanations, images))
        score["adcc_time"] = time() - previous_time

        return score

    def get_timm_model(self, model_id):
        timm_model = timm.create_model(model_id, in_chans=3, pretrained=True, checkpoint_path="")
        timm_model.eval()
        model_cfg = timm_model.default_cfg
        num_classes = model_cfg["num_classes"]
        if num_classes != 1000:
            pytest.skip(f"Model classes with more than 1000 classes are not supported yet")
        return timm_model, model_cfg

    def setup_dataset(self, model_name, dataset_parameters: list[tuple[Path, Path | None]]):

        # if model_name in IMAGENET_MODELS:
        #     if len(dataset_parameters) == 0:
        #         raise ValueError("Only one dataset is supported for ImageNet models")
        data_root, ann_path = dataset_parameters[0]

        self.dataset_type = define_dataset_type(data_root, ann_path)

        if self.dataset_type == DatasetType.COCO:
            self.dataset = datasets.CocoDetection(root=data_root, annFile=ann_path)
            self.dataset_labels_dict = {cats["id"]: cats["name"] for cats in self.dataset.coco.cats.values()}
            self.anns_to_gt_bboxes = coco_anns_to_gt_bboxes
        elif self.dataset_type == DatasetType.ILSVRC:
            self.dataset = CustomVOCDetection(root=data_root, download=False, year="2012", image_set="val")
            self.dataset_labels_dict = None
            self.anns_to_gt_bboxes = voc_anns_to_gt_bboxes