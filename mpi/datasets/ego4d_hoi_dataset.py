"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import json
import torch
import numpy as np
import pickle
import random

from PIL import Image, ImageOps
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from transformers import AutoTokenizer
import re
from collections import OrderedDict
from torchvision import transforms
import tqdm
import random
from torchvision.transforms import Compose, ConvertImageDtype, Lambda, \
                                   Normalize, Resize, ToTensor, \
                                   RandomResizedCrop, PILToTensor, ColorJitter, \
                                   RandomHorizontalFlip

import torchvision.transforms.functional as transF


def load_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def load_pickle(filename):
    with open(filename, mode="rb") as handle:
        data = pickle.load(handle)
        return data

def scaled_center_crop(target_resolution: int, frames) -> Image.Image:
    # Assert width >= height and height >= target_resolution
    orig_w, orig_h = frames[0].size
    assert orig_w >= orig_h >= target_resolution

    # Compute scale factor --> just a function of height and target_resolution
    scale_factor = target_resolution / orig_h
    for idx in range(len(frames)):
        frames[idx] = ImageOps.scale(frames[idx], factor=scale_factor)
        left = (frames[idx].size[0] - target_resolution) // 2
        frames[idx] = frames[idx].crop((left, 0, left + target_resolution, target_resolution))

    # Return "scaled and squared" images
    return frames


class Ego4DPretrainDataset():
    def __init__(self, train=True, pnr_only = False, post_only = False, data_ratio = 1.0, root_path = None):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """

        self.num_input_frames = 2
        self.num_target_frames = 1
        self.max_lang_length = 20 #32
        self.image_resulotion = 224
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        self.data_ratio = data_ratio
        self.pnrOnly = pnr_only
        self.postOnly = post_only
        self.threshold = 1 if pnr_only else 0 if post_only else 0.5


        assert not self.pnrOnly or not self.postOnly, "Can\'t set pnr_post and post_only at the same time."


        self.PIL_to_tensor = ToTensor()
        self.img_process = Compose(
                [
                    Resize((self.image_resulotion, self.image_resulotion), antialias=True),
                    ToTensor(),
                    Normalize(mean=mean, std=std),
                ])


        self.description = []
        self.frames = []
        self.boxes = []

        self.tokenizer = AutoTokenizer.from_pretrained('distilbert/distilbert-base-uncased')
        annotation_path = os.path.join(root_path, 'info_clips.json')
        annotation = load_json_file(annotation_path)
        

        for key in annotation.keys():
            if len(annotation[key]) == 0:
                continue
            for clip_info in annotation[key]:
                bbox_pnr = self.retrive_boxes(clip_info, key = 'pnr_frame')
                bbox_post = self.retrive_boxes(clip_info, key = 'post_frame')
                if len(bbox_pnr) == 0 or len(bbox_post) == 0:
                    continue
                bbox_pnr = bbox_pnr[0]
                bbox_post = bbox_post[0]
                self.boxes.append([bbox_pnr, bbox_post])
                self.description.append(clip_info['narration_text'].replace('#C C ', ''))
                frame_paths = [os.path.join(root_path, clip_info['pre_frame']['path']),
                            os.path.join(root_path, clip_info['pnr_frame']['path']),
                            os.path.join(root_path, clip_info['post_frame']['path'])]
                self.frames.append(frame_paths)

        if self.data_ratio < 1:
            random_idx = np.random.choice(np.arange(len(self.frames)), size=int(len(self.frames)*self.data_ratio), replace=False)
            self.description = [self.description[idx] for idx in random_idx]
            self.frames = [self.frames[idx] for idx in random_idx]
            self.boxes = [self.boxes[idx] for idx in random_idx]

        print('Number of clips:', len(self.description))

    def retrive_boxes(self, clip_info, key = 'pnr_frame'):
        boxes = []
        boxes_info = clip_info[key]['boxes']
        if len(boxes_info) == 0:
            return boxes
        for obj in boxes_info:
            if obj['object_type'] != 'object_of_change':
                continue
            boxes.append([obj['bbox'][key] for key in obj['bbox'].keys()])
        return boxes

    def crop_to_square(self, image_path, box_info):  
        image = Image.open(image_path[0]).convert("RGB")
        width, height = image.size
        min_side = min(width, height)
        
        for i in range(100):
            if min_side == height:
                left = random.randint(0, (width - min_side))
                right = left + height
                top = 0
                bottom = height
            else:
                left = 0
                right = width
                top = random.randint(0, (height - min_side))
                bottom = top + width

            # Do not crop boxes
            if (left > box_info[0] and left < box_info[0] + box_info[2]) \
                or (right > box_info[0] and right < box_info[0] + box_info[2]):
                continue
            else:
                break
        
        image = image.crop((left, top, right, bottom))

        cropped_imgs = [self.img_process(image)]

        for path in image_path[1:]:
            image = Image.open(path).convert("RGB")
            image = image.crop((left, top, right, bottom))
            cropped_imgs.append(self.img_process(image))
        
        return cropped_imgs, (left, top, right, bottom)


    def __getitem__(self, index):
        task_flag = random.random() > self.threshold # 1 for Post frame prediction, 0 for Pnr frame prediction
        # Get input frames
        # bbox_x, bbox_y, bbox_w, bbox_h = self.boxes[index][0 if task_flag else 1] # target frame: seen
        bbox_x, bbox_y, bbox_w, bbox_h = self.boxes[index][1 if task_flag else 0] # target frame: unseen 
        cropped_imgs, crop_info = self.crop_to_square(self.frames[index], box_info = (bbox_x, bbox_y, bbox_w, bbox_h))
        adj_images = torch.stack(cropped_imgs, dim=0)

        if task_flag:
            # Post frame
            input_frames  = adj_images[:2]
            target_frames = adj_images[2:]
        else:
            # Pnr frame
            input_frames  = torch.stack((adj_images[0], adj_images[2]))
            target_frames = adj_images[1].unsqueeze(0)
        
        # Normalize Bounding Boxes
        bbox_x = bbox_x - crop_info[0]
        bbox_y = bbox_y - crop_info[1]
        img_w = crop_info[2] - crop_info[0]
        img_h = crop_info[3] - crop_info[1]
        bbox_center_x = np.clip((bbox_x + bbox_w / 2), 0, img_w) / img_w
        bbox_center_y = np.clip((bbox_y + bbox_h / 2), 0, img_h) / img_h
        bbox_w = np.clip(bbox_w, 0, img_w) / img_w
        bbox_h = np.clip(bbox_h, 0, img_h) / img_h

        object_box = torch.tensor([1 - bbox_center_x, bbox_center_y, bbox_w, bbox_h])

        # Preprocess input language condition
        description = self.description[index]
        encoded_language = self.tokenizer(
            description, return_tensors="pt", max_length=self.max_lang_length, truncation=True, padding="max_length"
        )
        lang, lang_mask = encoded_language["input_ids"], encoded_language["attention_mask"]


        return input_frames, target_frames, lang, lang_mask, object_box, task_flag

    def __len__(self):
        return len(self.description)

    def collater(self, samples):

        image_list, question_list, answer_list, weight_list, time_list = [], [], [], [], []

        num_answers = []

        for sample in samples:
            image_list.append(sample["video_feature"])
            question_list.append(sample["question"])
            answers = sample["answer"]

            answer_list.extend(answers)
            num_answers.append(len(answers))
            time_list.append(sample["time"])

        return {
            "records": None,
            "vfeats": torch.stack(image_list, dim=0),
            "vfeat_lens": None,
            "word_ids": None,
            "char_ids": None,
            "s_labels": None,
            "e_labels": None,
            "h_labels": None,
            "questions": question_list,
            "answers": answer_list,
            "timesteps": torch.stack(time_list, dim=0),
        }