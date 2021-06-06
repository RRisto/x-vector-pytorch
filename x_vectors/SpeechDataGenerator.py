#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 14:09:31 2019

@author: Krishna
"""
import numpy as np
import torch
from utils import utils


class SpeechDataGenerator():
    """Speech dataset."""

    def __init__(self, manifest, mode, sr):
        """
        Read the textfile and get the paths
        """
        self.mode = mode
        self.sr = sr
        self.audio_links = [line.rstrip('\n').split(' ')[0] for line in open(manifest)]
        self.labels = [int(line.rstrip('\n').split(' ')[1]) for line in open(manifest)]

    def __len__(self):
        return len(self.audio_links)

    def __getitem__(self, idx):
        audio_link = self.audio_links[idx]
        class_id = self.labels[idx]
        spec = utils.load_data(audio_link, sr=self.sr, mode=self.mode)
        sample = {'features': torch.from_numpy(np.ascontiguousarray(spec)),
                  'labels': torch.from_numpy(np.ascontiguousarray(class_id))}
        return sample


class SpeechDataGeneratorFromFeat(SpeechDataGenerator):
    def __getitem__(self, idx):
        audio_link = self.audio_links[idx]
        class_id = self.labels[idx]
        spec = utils.load_npy_data(audio_link, mode=self.mode)
        sample = {'features': torch.from_numpy(np.ascontiguousarray(spec)),
                  'labels': torch.from_numpy(np.ascontiguousarray(class_id))}
        return sample


class SpeechDataGeneratorLive(SpeechDataGenerator):
    def __init__(self, audio_links, mode='test', sr=16000):
        """
        Read the textfile and get the paths
        """
        self.mode = mode
        self.sr = sr
        self.audio_links = audio_links
        self.labels = [0 for i in range(len(self.audio_links))]
