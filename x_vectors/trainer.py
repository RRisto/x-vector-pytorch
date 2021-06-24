#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 20:22:26 2020

@author: krishna
"""
from pprint import pprint

import torch
import pickle
import time
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader

from feature_extraction import extract_features
from x_vectors.SpeechDataGenerator import SpeechDataGeneratorLive, SpeechDataGeneratorFromFeat
import torch.nn as nn
import os
from torch import optim

from x_vectors.models.AdMSLoss import AdMSoftmaxLoss
from x_vectors.models.angleloss import AngleLoss
from pytorch_metric_learning import losses
from x_vectors.models.x_vector_Indian_LID import X_vector
from sklearn.metrics import accuracy_score
from utils.utils import speech_collate

torch.multiprocessing.set_sharing_strategy('file_system')


class Args():
    def __init__(self, training_filepath='meta_et_ru_fi/training.txt',
                 validation_filepath='meta_et_ru_fi/validation.txt', feature_link_suffix='_features.txt',
                 feature_folder='features', sr=16000, input_dim=257, num_classes=3, lamda_val=0.1, batch_size=256,
                 use_gpu=True, num_epochs=100, lr=0.001, weight_decay=0.0, betas=(0.9, 0.98), eps=1e-9,
                 loss_fun='CrossEntropyLoss', save_folder='save_model'):
        self.training_filepath = training_filepath
        self.validation_filepath = validation_filepath
        self.feature_link_suffix = feature_link_suffix
        self.feature_folder = feature_folder
        self.sr = sr
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.lamda_val = lamda_val
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        self.num_epochs = num_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.betas = betas
        self.eps = eps
        self.save_folder = save_folder
        assert loss_fun in ['CrossEntropyLoss', 'AngleLoss', 'AngularLoss', 'AdMSoftmaxLoss']
        self.loss_fun = loss_fun

    def get_args(self):
        pprint(vars(self))

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)


class Trainer():
    def __init__(self, args):
        self.args = args
        self.use_cuda = torch.cuda.is_available()
        self.device = "cuda" if self.use_cuda else "cpu"
        if self.args.loss_fun == 'CrossEntropyLoss':
            self.loss_fun = nn.CrossEntropyLoss()
        elif self.args.loss_fun == 'AngleLoss':
            self.loss_fun = AngleLoss()
        elif self.args.loss_fun == 'AngularLoss':
            self.loss_fun = losses.AngularLoss()
        elif self.args.loss_fun == 'AdMSoftmaxLoss':
            self.loss_fun = AdMSoftmaxLoss(3, args.num_classes, s=30.0, m=0.4, device=self.device)
        use_angular = self.args.loss_fun in ['AngleLoss']
        self.model = X_vector(args.input_dim, args.num_classes, use_angular=use_angular, device=self.device).to(
            self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                                    betas=args.betas, eps=args.eps)

    def _init_dls(self):
        self.dataloader_train = self._init_dl(self.args.training_filepath, 'train')
        # todo doesntwork in 'test' mode
        self.dataloader_val = self._init_dl(self.args.validation_filepath, 'train')

    def _init_dl(self, filepath, mode, shuffle=True):
        feature_links_filepath = self._create_features(filepath)
        dataset = SpeechDataGeneratorFromFeat(manifest=feature_links_filepath, mode=mode, sr=self.args.sr)
        dataloader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=shuffle,
                                collate_fn=speech_collate)
        return dataloader

    def _meta_path2feat_links_path(self, meta_filepath):
        meta_filepath = str(meta_filepath)
        orig_extension = os.path.splitext(meta_filepath)[-1]
        feature_links_filepath = meta_filepath.replace(orig_extension, self.args.feature_link_suffix)
        return feature_links_filepath

    def _create_features(self, filepath):
        feature_links_filepath = self._meta_path2feat_links_path(filepath)
        if os.path.isfile(feature_links_filepath):
            return feature_links_filepath

        print(f'Creating features for {filepath}')
        audio_links = [line.rstrip('\n').split(' ')[0] for line in open(filepath)]
        labels = [int(line.rstrip('\n').split(' ')[1]) for line in open(filepath)]

        with open(feature_links_filepath, 'a') as feat_link_file:
            for i, audio_file in enumerate(audio_links):
                extract_feats = extract_features(audio_file)
                filename_base = os.path.splitext(os.path.basename(audio_file))[0]
                dest_feature_filepath = f'{self.args.feature_folder}/{filename_base}.npy'
                np.save(dest_feature_filepath, extract_feats)
                feat_link_row = f'{dest_feature_filepath} {labels[i]}\n'
                feat_link_file.write(feat_link_row)
        return feature_links_filepath

    def train_epoch(self):
        train_loss_list = []
        full_preds = []
        full_gts = []
        self.model.train()
        self.dataloader_train = tqdm(self.dataloader_train)
        for i_batch, sample_batched in enumerate(self.dataloader_train):

            features = torch.from_numpy(
                np.asarray([torch_tensor.numpy().T for torch_tensor in sample_batched[0]])).float()
            labels = torch.from_numpy(np.asarray([torch_tensor[0].numpy() for torch_tensor in sample_batched[1]]))
            features, labels = features.to(self.device), labels.to(self.device)
            features.requires_grad = True
            self.optimizer.zero_grad()
            pred_logits, x_vec = self.model(features)
            loss = self.loss_fun(pred_logits, labels.type(torch.LongTensor).to(self.device))
            loss.backward()
            self.optimizer.step()
            train_loss_list.append(loss.item())

            if self.args.loss_fun in ['AngleLoss']:
                predictions = np.argmax(pred_logits[0].detach().cpu().numpy(), axis=1)
            else:
                predictions = np.argmax(pred_logits.detach().cpu().numpy(), axis=1)

            for pred in predictions:
                full_preds.append(pred)
            for lab in labels.detach().cpu().numpy():
                full_gts.append(lab)

        mean_acc = accuracy_score(full_gts, full_preds)
        mean_loss = np.mean(np.asarray(train_loss_list))
        return mean_acc, mean_loss

    def validate_epoch(self):
        self.model.eval()
        with torch.no_grad():
            val_loss_list = []
            full_preds = []
            full_gts = []
            self.dataloader_val = tqdm(self.dataloader_val)
            for i_batch, sample_batched in enumerate(self.dataloader_val):
                features = torch.from_numpy(
                    np.asarray([torch_tensor.numpy().T for torch_tensor in sample_batched[0]])).float()
                labels = torch.from_numpy(np.asarray([torch_tensor[0].numpy() for torch_tensor in sample_batched[1]]))
                features, labels = features.to(self.device), labels.to(self.device)
                pred_logits, x_vec = self.model(features)
                loss = self.loss_fun(pred_logits, labels.type(torch.LongTensor).to(self.device))
                val_loss_list.append(loss.item())
                if self.args.loss_fun in ['AngleLoss']:
                    predictions = np.argmax(pred_logits[0].detach().cpu().numpy(), axis=1)
                else:
                    predictions = np.argmax(pred_logits.detach().cpu().numpy(), axis=1)
                for pred in predictions:
                    full_preds.append(pred)
                for lab in labels.detach().cpu().numpy():
                    full_gts.append(lab)

            mean_acc = accuracy_score(full_gts, full_preds)
            mean_loss = np.mean(np.asarray(val_loss_list))
            return mean_acc, mean_loss

    def train(self, num_epochs=None):
        print('Initialising dataloaders')
        self._init_dls()
        if num_epochs is not None:
            self.args.num_epochs = num_epochs
        best_val_loss = np.inf
        for epoch in range(self.args.num_epochs):
            start_time = time.time()
            mean_train_acc, mean_train_loss = self.train_epoch()
            end_time = time.time()
            print(
                f'Total training loss {round(mean_train_loss, 3)} and training accuracy {round(mean_train_acc, 3)} after ' +
                f'{epoch} epochs, last epoch train time {round(end_time - start_time, 3)} sec')

            mean_val_acc, mean_val_loss = self.validate_epoch()
            print(
                f'Total validation loss {round(mean_val_loss, 3)} and validation accuracy {round(mean_val_acc, 3)} after {epoch} epochs')
            if best_val_loss > mean_val_loss:
                best_val_loss = mean_val_loss
                self.save_model(best_val_loss, epoch)

    def predict(self, filepaths):
        dataset = SpeechDataGeneratorLive(filepaths)
        dataloader = DataLoader(dataset, batch_size=self.args.batch_size, collate_fn=speech_collate)
        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            batch_logits = []
            batch_x_vecs = []
            for i_batch, sample_batched in enumerate(dataloader):
                features = torch.from_numpy(
                    np.asarray([torch_tensor.numpy().T for torch_tensor in sample_batched[0]])).float()
                labels = torch.from_numpy(np.asarray([torch_tensor[0].numpy() for torch_tensor in sample_batched[1]]))
                features, labels = features.to(self.device), labels.to(self.device)
                pred_logits, x_vec = self.model(features)
                batch_logits.extend(pred_logits)
                batch_x_vecs.extend(x_vec)
        return batch_logits, batch_x_vecs

    def save_model(self, mean_loss=None, epoch=None):
        args_save_path = os.path.join(self.args.save_folder, 'model_args.txt')
        self.args.save(args_save_path)
        model_save_path = os.path.join(self.args.save_folder, 'best_check_point_' + str(epoch) + '_' + str(mean_loss))
        state_dict = {'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(), 'epoch': epoch}
        torch.save(state_dict, model_save_path)
        print(f'saved best model to {model_save_path}')

    @classmethod
    def load_model(cls, args_file, state_file):
        args = Args.load(args_file)
        state_dict = torch.load(state_file)
        trainer = cls(args)
        trainer.model.load_state_dict(state_dict['model'])
        trainer.optimizer.load_state_dict(state_dict['optimizer'])
        return trainer
