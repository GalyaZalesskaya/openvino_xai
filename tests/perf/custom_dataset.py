from torchvision import datasets

# import collections
import os
# from pathlib import Path
# from typing import Any, Callable, Dict, List, Optional, Tuple, Union
# from xml.etree.ElementTree import Element as ET_Element

# try:
#     from defusedxml.ElementTree import parse as ET_parse
# except ImportError:
#     from xml.etree.ElementTree import parse as ET_parse

# from PIL import Image

# from .utils import download_and_extract_archive, verify_str_arg
# from .vision import VisionDataset

class CustomVOCDetection(datasets.VOCDetection):
    _SPLITS_DIR = "CLS-LOC"
    _TARGET_DIR = "Annotations"
    _TARGET_FILE_EXT = ".xml"

    def __init__(self, root, download=False, year="2012", image_set="val"):
        # Call the parent class's __init__ method
        try:
            super(CustomVOCDetection, self).__init__(root, year=year, image_set=image_set, download=download)
        except Exception:
            voc_root = root

            self.image_set = image_set

            splits_dir = os.path.join(voc_root, "ImageSets", self._SPLITS_DIR)
            split_f = os.path.join(splits_dir, image_set.rstrip("\n") + ".txt")
            with open(os.path.join(split_f)) as f:
                file_names = [x.split()[0] for x in f.readlines()]

            image_dir = os.path.join(voc_root, "Data", self._SPLITS_DIR, self.image_set)
            self.images = [os.path.join(image_dir, x + ".JPEG") for x in file_names]

            target_dir = os.path.join(voc_root, self._TARGET_DIR, self._SPLITS_DIR, self.image_set)
            self.targets = [os.path.join(target_dir, x + self._TARGET_FILE_EXT) for x in file_names]

            assert len(self.images) == len(self.targets)
