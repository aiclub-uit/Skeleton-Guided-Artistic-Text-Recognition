# Copyright (c) OpenMMLab. All rights reserved.
from .base_preprocessor import BasePreprocessor
from .tps_preprocessor import TPSPreprocessor
from .corner_preprocessor import CornerPreprocessor
from .skeleton_preprocessor import SkeletonPreprocessor
__all__ = ['BasePreprocessor', 'TPSPreprocessor', 'CornerPreprocessor', 'SkeletonPreprocessor']
