# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import Dataset
from params import Params
from de_distmult import DE_DistMult
from de_transe import DE_TransE
from de_simple import DE_SimplE
from tester import Tester


class Trainer:
    def __init__(self, dataset, params, model_name):
        instance_gen = globals()[model_name]
        self.model_name = model_name
        self.model = nn.DataParallel(instance_gen(dataset=dataset, params=params))
        self.dataset = dataset
        self.params = params

    def train(self, early_stop=False):
        self.model.train()
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.params.lr,
            weight_decay=self.params.reg_lambda
        )
        loss_f = nn.CrossEntropyLoss()

        start_all = time.time()  # Bắt đầu đo tổng thời gian train

        for epoch in range(1, self.params.ne + 1):
            last_batch = False
            total_loss = 0.0
            start_epoch = time.time()

            while not last_batch:
                optimizer.zero_grad()
                heads, rels, tails, years, months, days = self.dataset.nextBatch(self.params.bsize,
                                                                                 neg_ratio=self.params.neg_ratio)
                last_batch = self.dataset.wasLastBatch()

                scores = self.model(heads, rels, tails, years, months, days)

                num_examples = int(heads.shape[0] / (1 + self.params.neg_ratio))
                scores_reshaped = scores.view(num_examples, self.params.neg_ratio + 1)
                l = torch.zeros(num_examples).long().cuda()
                loss = loss_f(scores_reshaped, l)
                loss.backward()
                optimizer.step()
                total_loss += loss.cpu().item()

            print(f"Epoch {epoch} time: {time.time() - start_epoch:.2f} seconds")
            print(f"Loss in iteration {epoch}: {total_loss} ({self.model_name},{self.dataset.name})")

            if epoch % self.params.save_each == 0:
                self.saveModel(epoch)

        total_time = time.time() - start_all
        print(f"Total training time: {total_time:.2f} seconds")

    def saveModel(self, chkpnt):
        print("Saving the model")
        directory = "models/" + self.model_name + "/" + self.dataset.name + "/"
        if not os.path.exists(directory):
            os.makedirs(directory)

        torch.save(self.model, directory + self.params.str_() + "_" + str(chkpnt) + ".chkpnt")

