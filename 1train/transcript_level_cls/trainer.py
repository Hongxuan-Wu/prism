import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import os.path as osp
import sklearn.metrics
import torch
import torch.nn as nn
import sklearn
import numpy as np
import pandas as pd
import pdb

from torch import autocast
from transformers import AutoTokenizer

from model import TranscriptLevelCls
from dataloader import load_dataframe, make_loader, make_loader_predict
from utils import AverageMeter, to_gpu


def load_checkpoint(path, map_location, num_class):
    """Accept the newer head name without changing the public model or relaxing strict loading."""
    state = torch.load(path, map_location=map_location)
    mapped = {}
    for key, value in state.items():
        prefix = 'module.' if key.startswith('module.') else ''
        name = key[len(prefix):]
        if num_class > 2 and name.startswith('classifier.'):
            name = 'classification.' + name[len('classifier.'):]
        key = prefix + name
        if key in mapped:
            raise ValueError(f'Checkpoint contains conflicting classifier aliases: {key}')
        mapped[key] = value
    return mapped


class Trainer(object):
    def __init__(self, args, config):
        self.writer = args.writer
        self.logger = args.logger
        
        # load data
        if config['mode'] == 'test':
            self.df_val = load_dataframe(args, config)
        elif config['mode'] == 'predict':
            self.df_predict = load_dataframe(args, config)
        else:
            exit()
        
        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config['pretrained_path'],
            use_fast=True,
            trust_remote_code=True,
        )
        
        # set module
        self.model = TranscriptLevelCls(config).to(args.device)
        self.model = to_gpu(args, self.logger, self.model)

        # set loss
        if config['num_class'] == 2:
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        self.args = args
        self.config = config

    def test(self):
        data_loader = make_loader(self.args, self.config, self.df_val, self.tokenizer)
        check_point = load_checkpoint(os.path.join(self.config['checkpoint_dir'], "last.pth"), self.args.device, self.config['num_class'])
        
        self.model.load_state_dict(check_point)
        self.logger.info("Loaded model.")
        
        losses = AverageMeter()
        total_len = len(data_loader)
        predictions = []
        Y = []
        features = []
        
        for step, (X, mask, y) in enumerate(data_loader['val']):
            X = X.to(self.args.device)
            mask = mask.to(self.args.device)
            y = y.to(self.args.device)
            batch_size = y.size(0)

            with torch.no_grad():
                with autocast(device_type='cuda', dtype=torch.float16):
                    preds, feature = self.model(X, mask)
                    loss = self.criterion(preds, y if self.config['num_class'] == 2 else y.long())
            losses.update(loss.item(), batch_size)
            predictions.append(preds)
            Y.append(y)
            features.append(feature)

            if self.args.print_step:
                if step % self.args.display_step == 0 or step == (total_len - 1):
                    self.logger.info(f"Eval[{step}/{total_len}]  "
                        f"Loss: {losses.val:.5f} ({losses.avg:.5f})  "
                    )
        predictions = torch.cat(predictions)
        Y = torch.cat(Y)
        features = torch.cat(features)

        y_true = Y.cpu().numpy()
        y_pred = predictions.detach().cpu().numpy()
        features = features.detach().cpu().numpy()
        
        metrics = self.compute_metrics(y_true, y_pred)
        loss_avg = losses.avg
        
        return [loss_avg, metrics]
    
    def predict(self):
        data_loader = make_loader_predict(self.args, self.config, self.df_predict, self.tokenizer)
        
        check_point = load_checkpoint(os.path.join(self.config['checkpoint_dir'], "last.pth"), self.args.device, self.config['num_class'])
        self.model.load_state_dict(check_point)
        self.logger.info("Loaded model.")

        predictions = []
        
        for step, (X, mask) in enumerate(data_loader['predict']):
            X = X.to(self.args.device)
            mask = mask.to(self.args.device)

            with torch.no_grad():
                # with autocast(device_type='cuda', dtype=torch.float16):
                #     preds = self.model(X, mask)
                preds, feature = self.model(X, mask)
            predictions.append(preds)
        predictions = torch.cat(predictions)
        y_pred = predictions.detach().cpu().numpy()
        if self.config['num_class'] > 2:
            y_pred = y_pred.argmax(axis=1)
        df = pd.DataFrame(y_pred)
        
        filename = 'transcript_level_cls' + str(self.config['num_class'])
        if self.config['num_class'] == 2 and self.config['strong_weak']:
            filename += '_strong_weak'
        
        filename += '.csv'
        
        df.to_csv(osp.join(self.args.metrics_dir, filename), sep=',', index=False, header=False)

        if not os.path.exists(self.config['predict_dir']):
            os.makedirs(self.config['predict_dir'])
        
        df.to_csv(osp.join(self.config['predict_dir'], filename), sep=',', index=False, header=False)
    
    def compute_metrics(self, y_true, y_pred):
        if self.config['num_class'] == 2:
            y_pred = 1 / (1 + np.exp(-y_pred)) # sigmoid
            
            if len(np.unique(y_true)) == 2:
                fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true, y_pred)
                auc = sklearn.metrics.auc(fpr, tpr)
            else:
                # AUROC is undefined for a single-class subset, not zero or one.
                fpr, tpr, thresholds = np.array([]), np.array([]), np.array([])
                auc = float('nan')
            
            y_pred_class = np.round(y_pred)
            accuracy = sklearn.metrics.accuracy_score(y_true, y_pred_class)
            precision = sklearn.metrics.precision_score(y_true, y_pred_class, zero_division=0)
            recall = sklearn.metrics.recall_score(y_true, y_pred_class, zero_division=0)
            f1 = sklearn.metrics.f1_score(y_true, y_pred_class, zero_division=0)
            mcc = sklearn.metrics.matthews_corrcoef(y_true, y_pred_class)
            
            return {
                'fpr': fpr,
                'tpr': tpr,
                'thresholds': thresholds,
                'auc': auc, 
                'accuracy': accuracy, 
                'precision': precision, 
                'recall': recall,
                'f1': f1, 
                'mcc': mcc
            }
        else:
            y_pred = y_pred.argmax(axis=1)
        
            accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
            precision = sklearn.metrics.precision_score(y_true, y_pred, average="macro", zero_division=0)
            recall = sklearn.metrics.recall_score(y_true, y_pred, average="macro", zero_division=0)
            f1 = sklearn.metrics.f1_score(y_true, y_pred, average="macro", zero_division=0)
            mcc = sklearn.metrics.matthews_corrcoef(y_true, y_pred)

            return {
                'accuracy': accuracy, 
                'precision': precision, 
                'recall': recall, 
                'f1': f1, 
                'mcc': mcc 
            }
