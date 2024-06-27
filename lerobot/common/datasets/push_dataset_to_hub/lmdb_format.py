#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""
Contains utilities to process raw data format of HDF5 files like in: https://github.com/tonyzhaozh/act
"""

import gc
import shutil
from pathlib import Path

import lmdb
from pickle import loads
from torchvision.io import decode_jpeg
import numpy as np
import torch
import tqdm
from datasets import Dataset, Features, Image, Sequence, Value
from PIL import Image as PILImage

from lerobot.common.datasets.push_dataset_to_hub.utils import concatenate_episodes, save_images_concurrently
from lerobot.common.datasets.utils import (
    calculate_episode_data_index,
    hf_transform_to_torch,
)
from lerobot.common.datasets.video_utils import VideoFrame, encode_video_frames

def save_video(ep_idx, camera, videos_dir, imgs_array, fps, num_frames, ep_dict):
    img_key = f"observation.images.{camera}"
    tmp_imgs_dir = videos_dir / "tmp_images"
    save_images_concurrently(imgs_array, tmp_imgs_dir)

    # encode images to a mp4 video
    fname = f"{img_key}_episode_{ep_idx:06d}.mp4"
    video_path = videos_dir / fname
    encode_video_frames(tmp_imgs_dir, video_path, fps)

    # clean temporary images directory
    shutil.rmtree(tmp_imgs_dir)

    # store the reference to the video frame
    ep_dict[img_key] = [
        {"path": f"videos/{fname}", "timestamp": i / fps} for i in range(num_frames)
    ]

def string_to_utf8_array(s, length=72):
    # Encode the string to UTF-8
    utf8_encoded = s.encode('utf-8')
    
    # Convert to list of integers
    utf8_array = list(utf8_encoded)
    
    # Ensure the array length is exactly `length`
    if len(utf8_array) < length:
        # Pad with zeros if shorter
        utf8_array += [0] * (length - len(utf8_array))
    else:
        # Trim if longer
        utf8_array = utf8_array[:length]
    
    return utf8_array

def load_from_raw(raw_dir: Path, videos_dir: Path, fps: int, video: bool, episodes: list[int] | None = None):

    env = lmdb.open(str(raw_dir), readonly=True, create=False, lock=False)

    ep_dicts = []
    with env.begin() as txn:
        dataset_len = loads(txn.get('cur_step'.encode())) + 1
        inst_list = []
        inst_token_list = []
        rgb_static_list = []
        rgb_gripper_list = []
        state_list = []
        abs_action_list = []
        rel_action_list = []
        done_list = []
        last_ep_idx = loads(txn.get(f'cur_episode_{0}'.encode()))
        ep_start = 0
        for idx in range(dataset_len):
            ep_idx = loads(txn.get(f'cur_episode_{idx}'.encode()))
            if ep_idx == last_ep_idx + 100:
                print(f'{idx}/{dataset_len}')

                done_list[-1] = True
                num_frames = idx - ep_start
                ep_start = idx

                ep_dict = {}
                save_video(last_ep_idx, 'static', videos_dir, rgb_static_list, fps, num_frames, ep_dict)
                save_video(last_ep_idx, 'gripper', videos_dir, rgb_gripper_list, fps, num_frames, ep_dict)
                ep_dict["observation.state"] = torch.stack(state_list)
                ep_dict["inst"] = torch.stack(inst_list)
                ep_dict["inst_token"] = torch.stack(inst_token_list)
                ep_dict["action.abs"] = torch.stack(abs_action_list)
                ep_dict["action.rel"] = torch.stack(rel_action_list)
                ep_dict["episode_index"] = torch.tensor([ep_idx] * num_frames)
                ep_dict["frame_index"] = torch.arange(0, num_frames, 1)
                ep_dict["timestamp"] = torch.arange(0, num_frames, 1) / fps
                ep_dict["next.done"] = torch.tensor(done_list)
                ep_dicts.append(ep_dict)

                inst_list = []
                inst_token_list = []
                rgb_static_list = []
                rgb_gripper_list = []
                state_list = []
                abs_action_list = []
                rel_action_list = []
                done_list = []
                last_ep_idx = ep_idx 
            inst = torch.tensor(string_to_utf8_array(loads(txn.get(f'inst_{ep_idx}'.encode()))))
            inst_list.append(inst)
            inst_token_list.append(loads(txn.get(f'inst_token_{ep_idx}'.encode())))
            rgb_static_list.append(decode_jpeg(loads(txn.get(f'rgb_static_{idx}'.encode()))).permute(1, 2, 0).numpy())
            rgb_gripper_list.append(decode_jpeg(loads(txn.get(f'rgb_gripper_{idx}'.encode()))).permute(1, 2, 0).numpy())
            state_list.append(loads(txn.get(f'robot_obs_{idx}'.encode())))
            abs_action_list.append(loads(txn.get(f'abs_action_{idx}'.encode())))
            rel_action_list.append(loads(txn.get(f'rel_action_{idx}'.encode())))
            done_list.append(False)

        gc.collect()

    data_dict = concatenate_episodes(ep_dicts)

    total_frames = data_dict["frame_index"].shape[0]
    data_dict["index"] = torch.arange(0, total_frames, 1)
    return data_dict


def to_hf_dataset(data_dict, video) -> Dataset:
    features = {}

    keys = [key for key in data_dict if "observation.images." in key]
    for key in keys:
        if video:
            features[key] = VideoFrame()
        else:
            features[key] = Image()

    features["observation.state"] = Sequence(
        length=data_dict["observation.state"].shape[1], feature=Value(dtype="float32", id=None)
    )
    if "observation.velocity" in data_dict:
        features["observation.velocity"] = Sequence(
            length=data_dict["observation.velocity"].shape[1], feature=Value(dtype="float32", id=None)
        )
    if "observation.effort" in data_dict:
        features["observation.effort"] = Sequence(
            length=data_dict["observation.effort"].shape[1], feature=Value(dtype="float32", id=None)
        )
    features["action.abs"] = Sequence(
        length=data_dict["action.abs"].shape[1], feature=Value(dtype="float32", id=None)
    )
    features["action.rel"] = Sequence(
        length=data_dict["action.rel"].shape[1], feature=Value(dtype="float32", id=None)
    )
    features["inst"] = Sequence(
        length=data_dict["inst"].shape[1], feature=Value(dtype="float32", id=None)
    )
    features["inst_token"] = Sequence(
        length=data_dict["inst_token"].shape[1], feature=Value(dtype="float32", id=None)
    )
    features["episode_index"] = Value(dtype="int64", id=None)
    features["frame_index"] = Value(dtype="int64", id=None)
    features["timestamp"] = Value(dtype="float32", id=None)
    features["next.done"] = Value(dtype="bool", id=None)
    features["index"] = Value(dtype="int64", id=None)

    hf_dataset = Dataset.from_dict(data_dict, features=Features(features))
    hf_dataset.set_transform(hf_transform_to_torch)
    return hf_dataset


def from_raw_to_lerobot_format(
    raw_dir: Path,
    videos_dir: Path,
    fps: int | None = None,
    video: bool = True,
    episodes: list[int] | None = None,
):

    if fps is None:
        fps = 50

    data_dict = load_from_raw(raw_dir, videos_dir, fps, video, episodes)
    hf_dataset = to_hf_dataset(data_dict, video)
    episode_data_index = calculate_episode_data_index(hf_dataset)
    info = {
        "fps": fps,
        "video": video,
    }
    return hf_dataset, episode_data_index, info
