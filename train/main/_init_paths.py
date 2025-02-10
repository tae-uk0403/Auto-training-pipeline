# ------------------------------------------------------------------------------
# pose.pytorch
# Copyright (c) 2018-present Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import sys


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


this_dir = osp.dirname(__file__)

# train/ 디렉터리 (상위 1단계)
lib_path = os.path.abspath(os.path.join(this_dir, ".."))
add_path(lib_path)

mm_path = osp.join(this_dir, "..", "lib/poseeval/py-motmetrics")
add_path(mm_path)
