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
"""Tests for the rasterization_backend."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import tensorflow.compat.v2 as tf

from tensorflow_graphics.rendering import rasterization_backend
from tensorflow_graphics.util import test_case


class RasterizationBackendTest(test_case.TestCase):

  def _create_placeholders(self, shapes, dtypes):
    if tf.executing_eagerly():
      # If shapes is an empty list, we can continue with the test. If shapes
      # has None values, we shoud return.
      shapes = self._remove_dynamic_shapes(shapes)
      if shapes is None:
        return
    placeholders = self._create_placeholders_from_shapes(
        shapes=shapes, dtypes=dtypes)
    return placeholders

  @parameterized.parameters(
      (((7, 3), (5, 3), (4, 4)), (tf.float32, tf.int32, tf.float32),
       rasterization_backend.RasterizationBackends.OPENGL),)
  def test_rasterizer_rasterize_exception_not_raised(self, shapes, dtypes,
                                                     backend):
    """Tests that supported backends do not raise exceptions."""
    placeholders = self._create_placeholders(shapes, dtypes)
    try:
      rasterization_backend.rasterize(placeholders[0], placeholders[1],
                                      placeholders[2], (600, 800), backend)
    except Exception as e:  # pylint: disable=broad-except
      self.fail('Exception raised: %s' % str(e))

  @parameterized.parameters(
      (((7, 3), (5, 3), (4, 4)), (tf.float32, tf.int32, tf.float32), 'Foobar'),
      (((7, 3), (5, 3), (4, 4)), (tf.float32, tf.int32, tf.float32), 'Opengl'),
  )
  def test_rasterizer_rasterize_exception_raised(self, shapes, dtypes, backend):
    """Tests that unsupported backends raise exceptions."""
    placeholders = self._create_placeholders(shapes, dtypes)
    with self.assertRaisesRegexp(KeyError, backend):
      rasterization_backend.rasterize(placeholders[0], placeholders[1],
                                      placeholders[2], (600, 800), backend)


if __name__ == '__main__':
  test_case.main()
