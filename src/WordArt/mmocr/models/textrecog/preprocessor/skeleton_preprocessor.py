# Modified from https://github.com/clovaai/deep-text-recognition-benchmark
#
# Licensed under the Apache License, Version 2.0 (the "License");s
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from torchvision import transforms as T
from skeleton_tracing.swig import trace_skeleton
from skimage.morphology import skeletonize
from skimage.util import invert
from mmocr.utils.kmeans import clusterpixels

from mmocr.models.builder import PREPROCESSOR
from .base_preprocessor import BasePreprocessor

@PREPROCESSOR.register_module()
class SkeletonPreprocessor(BasePreprocessor):

    def __init__(self,
        algorithm="skeleton-tracing", # or skimage
        morphology=None, # list of 'open' or 'close'
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)

        self.algorithm = algorithm
        self.maxCorners = 200
        self.qualityLevel = 0.01
        self.minDistance = 3
        self.morphology = morphology
        self.max_size = 500
        print("PREPROCESS ALGORITHM:", algorithm)
        print("MORPHOLOGY:", morphology)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _segment_foreground_background(self, img):
        sin_img = img.transpose(1,2,0) * 255 # H, W, C
        gray_img = cv2.cvtColor(sin_img, cv2.COLOR_BGR2GRAY)
        gray_img = np.float32(gray_img)
        img_bg = np.zeros(gray_img.shape, dtype="uint8")
        clustering_output = clusterpixels(gray_img, 2)
        return (clustering_output, img_bg)

    def _skeleton_tracing_extract(self, clustering_output, img_bg):
        polys = trace_skeleton.from_numpy(clustering_output, 10, 999)
        for poly in polys:
            cv2.polylines(img_bg, np.int32([poly]), True, 1, 1)


        return img_bg

    def _skimage_skeleton_extraction(self, clustering_output, img_bg):
        skeleton = skeletonize(clustering_output)
        skeleton = skeleton.astype("uint8")
        return skeleton

    def _skeleton_enet(self, clustering_output, img_bg):
        clustering_output = clustering_output.astype(np.float32)
        img = self.enet_transform(clustering_output)
        img = img.unsqueeze(0).to(self.device)
        skeleton = self.enet(img).squeeze(0)

        # scale range [-1; 1] to range [0; 1]
        skeleton = (skeleton + 1) / 2
        return skeleton.squeeze(0).cpu().detach().numpy()

    def _resize_image_if_large(self, image, max_size):
        # Get the dimensions of the image
        height, width = image.shape[:2]

        # Check if either dimension exceeds the max_size
        if max(width, height) > max_size:
            # Calculate the scaling factor to maintain the aspect ratio
            scaling_factor = max_size / max(width, height)

            # Calculate new dimensions
            new_width = int(width * scaling_factor)
            new_height = int(height * scaling_factor)

            # Resize the image while maintaining aspect ratio
            resized_image = cv2.resize(
                image, (new_width, new_height), interpolation=cv2.INTER_AREA
            )
            return resized_image
        else:
            # If no resizing is needed, return the original image
            return image

    def _get_relative_kernel_size(self, image, min_kernel_size=3, max_kernel_size=9):
        # Get the dimensions of the image
        height, width = image.shape[:2]

        # Choose a scaling factor based on the size of the image
        # Larger images will get a larger kernel size
        # Let's assume images smaller than 300x300 use the smallest kernel (e.g., 3x3), and we scale up from there
        scaling_factor = min(width, height) / 300.0

        # Calculate the kernel size and round it to the nearest odd number (kernels need to be odd-sized)
        kernel_size = int(3 * scaling_factor)
        kernel_size = (
            kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        )  # Ensure it's odd

        # Cap the kernel size to max_kernel_size (e.g., 9x9)
        kernel_size = max(min_kernel_size, min(kernel_size, max_kernel_size))

        return (kernel_size, kernel_size)


    def _perform_morphology(self, img):
        if self.morphology is None:
            raise ValueError("Morphology is None, cannot perform morphology transform")

        img = self._resize_image_if_large(img, self.max_size)
        kernel = self._get_relative_kernel_size(img)

        # print(img.shape, img.dtype)
        img = img.astype(np.uint8)

        for morp in self.morphology:
            cv2_morp = cv2.MORPH_CLOSE if morp == "close" else cv2.MORPH_OPEN
            img = cv2.morphologyEx(img, cv2_morp, kernel)

        return img

    def forward(self, batch_img):
        """
        Args:
            batch_img (Tensor): Images to be rectified with size
                :math:`(N, C, H, W)`.

        Returns:
            Tensor: Skeleton map map with size :math:`(N, 1, H, W)`.
        """
        device = batch_img.device
        img_np = batch_img.cpu().numpy()

        batch_skeleton_map = torch.Tensor()
        for i in range(img_np.shape[0]):
            skeleton_img = None

            clustering_output, img_bg = self._segment_foreground_background(img_np[i])

            if self.morphology is not None:
                clustering_output = self._perform_morphology(clustering_output)

            if self.algorithm == "skeleton-tracing":
                skeleton_img = self._skeleton_tracing_extract(clustering_output, img_bg)
            elif self.algorithm == "skimage":
                skeleton_img = self._skimage_skeleton_extraction(clustering_output, img_bg)
            else:
                raise NotImplementedError


            skeleton_map = torch.tensor(skeleton_img).unsqueeze(0).unsqueeze(0)
            skeleton_map = skeleton_map.to(torch.float32)
            batch_skeleton_map = torch.cat([batch_skeleton_map, skeleton_map], dim=0)

        batch_skeleton_map = batch_skeleton_map.to(device)

        return batch_skeleton_map
