import os
import time
import torch
import torch.nn as nn
from dataset import Dataset
from params import Params
from de_distmult import DE_DistMult
from de_transe import DE_TransE
from de_simple import DE_SimplE
from tester import Tester


class Trainer:
    def __init__(self, dataset, params, model_name, resume_from=None):
        instance_gen = globals()[model_name]
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = nn.DataParallel(instance_gen(dataset=dataset, params=params)).to(self.device)
        self.dataset = dataset
        self.params = params

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.params.lr,
            weight_decay=self.params.reg_lambda
        )
        self.start_epoch = 1  # Mặc định train từ epoch 1

        if resume_from is not None:
            print(f"🔄 Loading checkpoint from: {resume_from}")
            checkpoint = torch.load(resume_from, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.start_epoch = checkpoint['epoch'] + 1  # Tiếp tục từ epoch kế tiếp

    def train(self, early_stop=False):
        self.model.train()
        loss_f = nn.CrossEntropyLoss()
        start_all = time.time()

        log_dir = f"models/{self.model_name}/{self.dataset.name}/"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_path = os.path.join(log_dir, "train_times.log")

        with open(log_path, "a") as log_file:
            log_file.write(f"Training started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

            for epoch in range(self.start_epoch, self.params.ne + 1):
                last_batch = False
                total_loss = 0.0
                start_epoch = time.time()

                while not last_batch:
                    self.optimizer.zero_grad()
                    heads, rels, tails, years, months, days = self.dataset.nextBatch(
                        self.params.bsize,
                        neg_ratio=self.params.neg_ratio
                    )
                    last_batch = self.dataset.wasLastBatch()

                    # Chuyển dữ liệu sang device
                    heads = heads.to(self.device)
                    rels = rels.to(self.device)
                    tails = tails.to(self.device)
                    years = years.to(self.device)
                    months = months.to(self.device)
                    days = days.to(self.device)

                    scores = self.model(heads, rels, tails, years, months, days)

                    num_examples = int(heads.shape[0] / (1 + self.params.neg_ratio))
                    scores_reshaped = scores.view(num_examples, self.params.neg_ratio + 1)
                    l = torch.zeros(num_examples).long().to(self.device)
                    loss = loss_f(scores_reshaped, l)
                    loss.backward()
                    self.optimizer.step()
                    total_loss += loss.item()

                epoch_time = time.time() - start_epoch
                print(f"Epoch {epoch} time: {epoch_time:.2f} seconds")
                print(f"Loss in iteration {epoch}: {total_loss} ({self.model_name},{self.dataset.name})")

                log_file.write(f"Epoch {epoch} time: {epoch_time:.2f} seconds, loss: {total_loss:.4f}\n")

                if epoch % self.params.save_each == 0:
                    self.saveModel(epoch)

            total_time = time.time() - start_all
            print(f"Total training time: {total_time:.2f} seconds")
            log_file.write(f"Training finished at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_file.write(f"Total training time: {total_time:.2f} seconds\n\n")

    def saveModel(self, chkpnt):
        print("💾 Saving the model at epoch", chkpnt)
        directory = f"models/{self.model_name}/{self.dataset.name}/"
        if not os.path.exists(directory):
            os.makedirs(directory)

        save_path = directory + self.params.str_() + f"_{chkpnt}.chkpnt"
        torch.save({
            'epoch': chkpnt,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, save_path)
