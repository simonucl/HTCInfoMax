#!/usr/bin/env python
# coding:utf-8

import helper.logger as logger
from train_modules.evaluation_metrics import evaluate
import torch
import tqdm
import numpy as np

class Trainer(object):
    def __init__(self, model, criterion, optimizer, scheduler, vocab, config):
        """
        :param model: Computational Graph
        :param criterion: train_modules.ClassificationLoss object
        :param optimizer: optimization function for backward pass
        :param vocab: vocab.v2i -> Dict{'token': Dict{vocabulary to id map}, 'label': Dict{vocabulary
        to id map}}, vocab.i2v -> Dict{'token': Dict{id to vocabulary map}, 'label': Dict{id to vocabulary map}}
        :param config: helper.Configure object
        """
        super(Trainer, self).__init__()
        self.model = model
        self.vocab = vocab
        self.config = config
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

    def update_lr(self):
        """
        (callback function) update learning rate according to the decay weight
        """
        logger.warning('Learning rate update {}--->{}'
                       .format(self.optimizer.param_groups[0]['lr'],
                               self.optimizer.param_groups[0]['lr'] * self.config.train.optimizer.lr_decay))
        for param in self.optimizer.param_groups:
            param['lr'] = self.config.train.optimizer.learning_rate * self.config.train.optimizer.lr_decay

    def run(self, data_loader, epoch, stage, mode='TRAIN'):
        """
        training epoch
        :param data_loader: Iteration[Dict{'token':tensor, 'label':tensor, }]
        :param epoch: int, log results of this epoch
        :param stage: str, e.g. 'TRAIN'/'DEV'/'TEST', figure out the corpus
        :param mode: str, ['TRAIN', 'EVAL'], train with backward pass while eval without it
        :return: metrics -> {'precision': 0.xx, 'recall': 0.xx, 'micro-f1': 0.xx, 'macro-f1': 0.xx}
        """
        predict_probs = []
        target_labels = []
        total_loss = 0.0
        num_batch = data_loader.__len__()

        for batch in tqdm.tqdm(data_loader):
            text_label_mi_disc_loss, label_prior_loss, logits, loss_weight = self.model(batch)
            if self.config.train.loss.recursive_regularization.flag:
                recursive_constrained_params = self.model.htcinfomax.linear.weight
            else:
                recursive_constrained_params = None
            loss_predictor = self.criterion(logits,
                                  batch['label'].to(self.config.train.device_setting.device),
                                  recursive_constrained_params)
            
            print('classifier loss: ', loss_predictor)
            print('text_label_mi_disc_loss: ', text_label_mi_disc_loss)
            print('label_prior_loss: ', label_prior_loss)

            loss = loss_predictor + loss_weight*text_label_mi_disc_loss + (1-loss_weight)*label_prior_loss
            # loss = loss_predictor
            total_loss += loss.item()
            print('loss weight: ', loss_weight)
            print('loss: ', loss
                  )
            if mode == 'TRAIN':
                if "bert" in self.config.model.type:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1, norm_type=2)
                    # self.scheduler.step()
                    
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            predict_results = torch.sigmoid(logits).cpu().tolist()
            predict_probs.extend(predict_results)
            target_labels.extend(batch['label_list'])
            print(np.where(np.array(predict_results[0]) > 0.5))
            print(np.array(batch['label_list'][0]))
            # print(np.where(np.array(batch['label_list'][0]) > 0.5))

        total_loss = total_loss / num_batch
        if mode == 'EVAL':
            metrics = evaluate(predict_probs,
                               target_labels,
                               self.vocab,
                               self.config.eval.threshold)
            # metrics = {'precision': precision_micro,
            #             'recall': recall_micro,
            #             'micro_f1': micro_f1,
            #             'macro_f1': macro_f1}
            logger.info("%s performance at epoch %d --- Precision: %f, "
                        "Recall: %f, Micro-F1: %f, Macro-F1: %f, Loss: %f.\n"
                        % (stage, epoch,
                           metrics['precision'], metrics['recall'], metrics['micro_f1'], metrics['macro_f1'],
                           total_loss))
            return metrics

    def train(self, data_loader, epoch):
        """
        training module
        :param data_loader: Iteration[Dict{'token':tensor, 'label':tensor, }]
        :param epoch: int, log results of this epoch
        :return: metrics -> {'precision': 0.xx, 'recall': 0.xx, 'micro-f1': 0.xx, 'macro-f1': 0.xx}
        """
        self.model.train()
        return self.run(data_loader, epoch, 'Train', mode='TRAIN')

    def eval(self, data_loader, epoch, stage):
        """
        evaluation module
        :param data_loader: Iteration[Dict{'token':tensor, 'label':tensor, }]
        :param epoch: int, log results of this epoch
        :param stage: str, TRAIN/DEV/TEST, log the result of the according corpus
        :return: metrics -> {'precision': 0.xx, 'recall': 0.xx, 'micro-f1': 0.xx, 'macro-f1': 0.xx}
        """
        self.model.eval()
        return self.run(data_loader, epoch, stage, mode='EVAL')
