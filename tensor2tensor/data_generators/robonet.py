# coding=utf-8
# Copyright 2019 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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

"""Berkeley (BAIR) robot pushing dataset.

Self-Supervised Visual Planning with Temporal Skip Connections
Frederik Ebert, Chelsea Finn, Alex X. Lee, and Sergey Levine.
https://arxiv.org/abs/1710.05268

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import video_utils
from tensor2tensor.layers import modalities
from tensor2tensor.utils import registry
import h5py
import tensorflow as tf


from robonet.datasets import load_metadata
from robonet.datasets.robonet_dataset import RoboNetDataset
import tensorflow as tf
sess = tf.Session()
  
hparams = {'RNG': 0, 'ret_fnames': True, 'load_T': 30, 'load_random_cam':True, 'sub_batch_size': 1, 'action_mismatch': 3, 'state_mismatch': 3, 'splits':[0.8, 0.1, 0.1], 'same_cam_across_sub_batch':True}

  
all_robonet = load_metadata('/iris/u/surajn/data/robonet/hdf5')
database = all_robonet[all_robonet['robot'] == 'sawyer']
database = database[database['adim'] == 4]
data = RoboNetDataset(batch_size=1, dataset_files_or_metadata=database, hparams=hparams)
train_images = tf.reshape(
                      tf.image.resize(
                        tf.reshape(data['images', 'train'], [1*30, 48, 64, 3]), size=(64, 64))
                  , [1, 30, 64, 64, 3]) * 255.0
train_actions = data['actions', 'train']
test_images = tf.reshape(
                      tf.image.resize(
                        tf.reshape(data['images', 'test'], [1*30, 48, 64, 3]), size=(64, 64))
                  , [1, 30, 64, 64, 3]) * 255.0
test_actions = data['actions', 'test']

# DATA_URL = ("/cvgl2/u/surajn/data/sv2p_train_data.hdf5")
NUMEP = 10000
EPLEN = 30

@registry.register_problem
class Robonet(video_utils.VideoProblem):

  @property
  def num_channels(self):
    return 3

  @property
  def frame_height(self):
    return 64

  @property
  def frame_width(self):
    return 64

  @property
  def is_generate_per_split(self):
    return True

  # num_train_files * num_videos * num_frames
  @property
  def total_number_of_frames(self):
    return 300000

  def max_frames_per_video(self, hparams):
    return 30

  @property
  def random_skip(self):
    return False

  @property
  def only_keep_videos_from_0th_frame(self):
    return False

  @property
  def use_not_breaking_batching(self):
    return True

  @property
  def extra_reading_spec(self):
    """Additional data fields to store on disk and their decoders."""
    data_fields = {
        "frame_number": tf.FixedLenFeature([1], tf.int64),
        "action":tf.FixedLenFeature([4], tf.float32),
    }
    decoders = {
        "frame_number": tf.contrib.slim.tfexample_decoder.Tensor(
            tensor_key="frame_number"),
        "action": tf.contrib.slim.tfexample_decoder.Tensor(tensor_key="action"),
    }
    return data_fields, decoders

  def hparams(self, defaults, unused_model_hparams):
    p = defaults
    p.modality = {"inputs": modalities.ModalityType.VIDEO,
                  "action":modalities.ModalityType.REAL_L2_LOSS,
                  "targets": modalities.ModalityType.VIDEO}
    p.vocab_size = {"inputs": 256,
                    "action":4,
                    "targets": 256}

  def parse_frames(self, dataset_split):
    for ep in range(start_ep, end_ep):
      if dataset_split == problem.DatasetSplit.TRAIN:
          ims = sess.run(train_images)
          acts = sess.run(train_actions)
      else:
          ims = sess.run(test_images)
          acts = sess.run(test_actions)
      for step in range(EPLEN):
          frame = ims[0,step]
          action = acts[0, step]
          yield step, frame, action
            
  def generate_samples(self, data_dir, tmp_dir, dataset_split):
#     path = generator_utils.maybe_download(
#         tmp_dir, os.path.basename(DATA_URL), DATA_URL)
    path= DATA_URL

    f = h5py.File(path, "r")

    for frame_number, frame, action in self.parse_frames(dataset_split):
      yield {
          "frame_number": [frame_number],
          "frame": frame,
          "action": action.tolist(),
      }