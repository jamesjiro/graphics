# Copyright 2020 The TensorFlow Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Lint as: python3
"""Tests for tensorflow_graphics.datasets.features.pose_feature."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow_graphics.geometry.transformation import rotation_matrix_3d
from tensorflow_graphics.datasets.features import pose_feature
import tensorflow_datasets as tfds


class PoseFeatureTest(tfds.testing.FeatureExpectationsTestCase):

  def test_pose_feature(self):
    expected_rotation = np.eye(3)
    expected_translation = np.zeros(3)

    expected_pose = {'R': expected_rotation.astype(np.float32), 't': expected_translation.astype(np.float32)}

    self.assertFeature(
        feature=pose_feature.Pose(),
        shape={
            'R': (3, 3),
            't': (3,)
        },
        dtype={
            'R': tf.float32,
            't': tf.float32
        },
        tests=[
            # FeaturesDict
            tfds.testing.FeatureExpectationItem(
                value=expected_pose,
                expected=expected_pose,
            ),
        ],
    )


if __name__ == '__main__':
  tfds.testing.test_main()
