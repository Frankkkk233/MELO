import copy
import random
import importlib
import logging
from time import time
import hydra
from omegaconf import OmegaConf,open_dict
import numpy as np
import torch
from utils import *
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
import json
import os
LOG = logging.getLogger(__name__)

class scotus_trainer:
    def __init__(self, config, alg, tokenize, metric, edit_loader, upstream_loader):
        self.config = config
        self.alg = alg
        self.tokenize = tokenize
        self.metric = metric
        self.edit_loader = edit_loader
        self.upstream_loader  = upstream_loader
        self.batch_size = config.grace.num_edit_per_block
        pass
    def pre_editing_analyse(self):
        self.alg.disable_melo()
        with torch.no_grad():
            metric_list = []
            for batch in iter(self.edit_loader):
                edit_input = self.tokenize(batch, self.alg.model_tok, self.config['device'])
                metric_list.append(self.metric(self.alg, edit_input))

            original_edits = torch.Tensor(metric_list).nanmean()
            LOG.info(f'Average performance on edit set: {original_edits}')

            TRR = []
            for u in iter(self.upstream_loader):
                upstream_input = self.tokenize(u, self.alg.model_tok, self.config['device'])
                TRR.append(self.metric(self.alg, upstream_input))

            TRR = torch.Tensor(TRR).nanmean()
            LOG.info(f"Orignial TRR: {TRR}")

    def run_edit(self):
        self.alg.enable_melo()
        n_edits = 0
        batch_history = []
        total_edit_time = 0
        all_edit_time = {}
        all_HIS = {}
        all_UP = {}
        all_ARR = {}
        for i, batch in tqdm(enumerate(self.edit_loader)):

            tokens = self.tokenize(batch, self.alg.model_tok, self.config["device"])

            print(n_edits)

            # --- Check that the model is actually making a mistake (for detecting hallucination, `is_error` always returns True) or stop after making enough edits ---
            if n_edits <= self.config.max_n_edits:
                LOG.info(f' ----------------------   Edit Batch {i} -----------------------------')
                n_edits += self.batch_size
                batch_history.append(tokens)  # Append new batch to growing history of edits
                # --- perform edit ---
                edit_start = time()
                self.alg.edit(tokens)
                edit_time = time() - edit_start
                total_edit_time += edit_time

                # --- Compute and log metrics ---
                log_dict = {}
                with torch.no_grad():
                    ES = self.metric(self.alg, tokens)
                    LOG.info(f"[+edit results+] Current Batch Accuracy: {ES}")
                    if (n_edits > 0 and n_edits % self.config.grace.metric_period == 0) or (
                            i == len(self.edit_loader) - 1):  # Compute historical metrics every k edits to save time



                        metric_list = [self.metric(self.alg, tokens) for tokens in batch_history]
                        ERR = torch.tensor(metric_list).nanmean()

                        metric_list = [self.metric(self.alg, self.tokenize(batch, self.alg.model_tok, self.config["device"], test=True)) for batch
                                       in iter(self.upstream_loader)]
                        TRR = torch.tensor(metric_list)
                        TRR = torch.Tensor(TRR).nanmean()

                        # --- Log metrics and push to Weights & Biases ---
                        log_dict["TRR"] = {'UP': TRR.item()}  # Test Retention Rate
                        log_dict["ERR"] = {'HIS': ERR.item()}  # Error Retention Rate
                        log_dict["ES"] = ES  # Edit Success
                        log_dict["train_time"] = edit_time / 60  # Time it takes to make one edit

                        log_dict["n_edits"] = n_edits  # Raw edit label



                        LOG.info(f"Number of edits: {n_edits}")
                        for k in log_dict:
                            LOG.info(f"[+edit results+]{k}: {log_dict[k]}")
                        all_UP[n_edits] = log_dict["TRR"]
                        all_HIS[n_edits] = log_dict["ERR"]

                        all_edit_time[n_edits] = total_edit_time

        with open(f'log.pkl', 'wb') as f:
            pickle.dump(
                {'all_UP': all_UP, 'all_HIS': all_HIS, 'all_edit_time': all_edit_time}, f)
        LOG.info(f"[**Total Edit Time**] {total_edit_time / 60} mins")

class zsre_trainer:
    def __init__(self, config, alg, tokenize, metric, edit_loader, upstream_loader, edit_holdout_loader):
        self.config = config
        self.alg = alg
        self.tokenize = tokenize
        self.metric = metric
        self.edit_loader = edit_loader
        self.upstream_loader  = upstream_loader
        self.edit_holdout_loader = edit_holdout_loader
        self.batch_size = config.grace.num_edit_per_block

    def pre_editing_analyse(self,all_metrics):
        self.alg.disable_melo()
        with torch.no_grad():
            for batch in tqdm(iter(self.edit_holdout_loader),desc='pre:',total=len(self.edit_holdout_loader)):
                metrics = {"pre": self.metric(self.alg.model.model, self.config.model.name, self.alg.model_tok, batch, self.config['device'])}
                all_metrics.append(metrics)
        return all_metrics

    def run_edit(self):
        # --- editing start ---
        all_metrics = []
        all_metrics = self.pre_editing_analyse(all_metrics)
        self.alg.enable_melo()
        n_edits = 0
        batch_history = []
        total_edit_time = 0
        all_edit_time = {}
        all_HIS = {}
        all_HOLDOUT = {}
        all_UP = {}
        all_VecDB = {}
        def edit_evaluation(all_metrics, request, edited_model, idx):
            all_metrics[idx].update({
                'case_id': idx,
                "requested_rewrite": request,
                "post": self.metric(edited_model, self.config.model.name, self.alg.model_tok, request, self.config['device'], eval_metric='exact match', test_generation=True),
            })
            
            if 'locality' in all_metrics[idx]['post'].keys() and len(all_metrics[idx]['post']['locality']) > 0:
                for locality_key in request['locality'].keys():
                    locality_result = []
                    for ans, label in zip(all_metrics[idx]['post']['locality'][f'{locality_key}_output'], all_metrics[idx]['pre']['locality'][f'{locality_key}_output']):
                        #此label是pre得预测token，不是loc_ans
                        locality_result.append(np.mean(np.equal(ans, label)))
                    all_metrics[idx]['post']['locality'][f'{locality_key}_acc'] = locality_result
                    all_metrics[idx]['post']['locality'].pop(f'{locality_key}_output')
                all_metrics[idx]['pre'].pop('locality')

        for i, batch in tqdm(enumerate(self.edit_loader)):
            if i == 25:
                print(i)
            LOG.info(f'-------------------------    Edit Batch {i} ----------------------------------')
            tokens = self.tokenize(batch, self.alg.model_tok, self.config['device'])
            if n_edits < self.config.max_n_edits:
                n_edits += self.batch_size
                batch_history.append(tokens)

                # --- perform edit ---
                edit_start = time()
                self.alg.edit(tokens)
                edit_time = time() - edit_start
                total_edit_time += edit_time

                # --- Compute and log metrics ---
                log_dict = {}
                with torch.no_grad():
                    edit_info = self.metric(self.alg.model.model,self.config.model.class_name,self.alg.model_tok,batch,self.config.device)
                    LOG.info(f'Batch {i} after Editing:')
                    mean_rewrite = sum(edit_info['rewrite_acc']) / len(edit_info['rewrite_acc'])
                    mean_rephrase = sum(edit_info['rephrase_acc']) / len(edit_info['rephrase_acc'])
                    print('rewrite_acc:', edit_info['rewrite_acc'])
                    print('rephrase_acc:', edit_info['rephrase_acc'])
                    print('mean_rewrite_acc:', mean_rewrite)
                    print('mean_rephrase_acc:', mean_rephrase)

                    if (i > 0 and n_edits % self.config.grace.metric_period == 0) or (i == len(self.edit_loader) - 1):
       
                        LOG.info(
                            f'-------------------------    Eval all {n_edits} history edits----------------------------------')


                        if self.config.task == 'qa':
                            for i, request in enumerate(tqdm(self.edit_holdout_loader,desc='post', total=len(self.edit_holdout_loader))):
                                edit_evaluation(all_metrics, request, self.alg.model.model, i)
                        
                        # 创建logs目录（如果不存在）
                        os.makedirs('./logs', exist_ok=True)
                        for metric in all_metrics:
                            if 'requested_rewrite' in metric:
                                del metric['requested_rewrite']
                        
                        json.dump(all_metrics, open('./logs/'+self.config.model.class_name+'_'+self.config.task+'.json', 'w+'), indent=4)
                        all_edit_time[n_edits] = total_edit_time
                        VecDB_info = self.alg.get_VecDB_info()
                        for k in VecDB_info:
                            LOG.info(f"[+VecDB Info+]{k}: {VecDB_info[k]}")
                        all_VecDB[n_edits] = VecDB_info
                        pass

        with open(f'log.pkl', 'wb') as f:
            pickle.dump(
                {'all_UP': all_UP, 'all_HIS': all_HIS, 'all_HOLDOUT': all_HOLDOUT, 'all_edit_time': all_edit_time,
                 'all_VecDB': all_VecDB}, f)

        LOG.info(f"[**Total Edit Time**] {total_edit_time / 60} mins")

class hallucination_trainer:
    def __init__(self, config, alg, tokenize, metric, edit_loader, upstream_loader, accurate_loader):
        self.config = config
        self.alg = alg
        self.tokenize = tokenize
        self.metric = metric
        self.edit_loader = edit_loader
        self.upstream_loader  = upstream_loader
        self.accurate_loader = accurate_loader
        self.batch_size = config.grace.num_edit_per_block

    def pre_editing_analyse(self):
        self.alg.disable_melo()

        with torch.no_grad():
            metric_list = []
            for batch in iter(self.edit_loader):
                edit_input = self.tokenize(batch, self.alg.model_tok, self.config['device'])
                metric_list.append(self.metric(self.alg, edit_input))

            original_edits = torch.Tensor(metric_list).nanmean()
            LOG.info(f'Average performance on edit set: {original_edits}')

            ARR = torch.tensor([self.metric(self.alg, self.tokenize(batch, self.alg.model_tok, self.config["device"], test=True))
                                for batch in iter(self.accurate_loader)]).nanmean() # Log first PPL before edits
            LOG.info(f"Original Accurate: {ARR}")

            TRR = []
            for u in iter(self.upstream_loader):
                upstream_input = self.tokenize(u, self.alg.model_tok, self.config['device'])
                TRR.append(self.metric(self.alg, upstream_input))

            TRR = torch.tensor(TRR)
            # TRR = torch.tensor([metric(editor, tokenize(e, editor.tokenizer, config["device"], test=True)) for e in iter(upstream_loader)])
            # TRR = TRR[~torch.isnan(TRR)]
            TRR = TRR.nanmean()
            print("Orignial TRR: ", TRR)
            LOG.info(f"Orignial TRR: {TRR}")

    def run_edit(self):
        self.alg.enable_melo()
        n_edits = 0
        batch_history = []
        total_edit_time = 0
        all_edit_time = {}
        all_HIS = {}
        all_UP = {}
        all_ARR = {}
        for i, batch in tqdm(enumerate(self.edit_loader)):

            tokens = self.tokenize(batch, self.alg.model_tok, self.config["device"])

            print(n_edits)

            # --- Check that the model is actually making a mistake (for detecting hallucination, `is_error` always returns True) or stop after making enough edits ---
            if n_edits <= self.config.max_n_edits:
                LOG.info(f' ----------------------   Edit Batch {i} -----------------------------')
                n_edits += self.batch_size

                batch_history.append(tokens)  # Append new batch to growing history of edits


                # --- perform edit ---
                edit_start = time()
                self.alg.edit(tokens)
                edit_time = time() - edit_start
                total_edit_time += edit_time

                # --- Compute and log metrics ---
                log_dict = {}
                with torch.no_grad():
                    ES = self.metric(self.alg, tokens)
                    LOG.info(f"[+edit results+] Current Batch PPL: {ES}")
                    if (i > 0 and n_edits % self.config.grace.metric_period == 0) or (
                            i == len(self.edit_loader) - 1):  # Compute historical metrics every k edits to save time

                        ARR = torch.tensor([self.metric(self.alg, self.tokenize(batch, self.alg.model_tok, self.config["device"])) for batch in
                                            iter(self.accurate_loader)]).nanmean()

                        metric_list = [self.metric(self.alg, tokens) for tokens in batch_history]
                        ERR = torch.tensor(metric_list).nanmean()

                        metric_list = [self.metric(self.alg, self.tokenize(batch, self.alg.model_tok, self.config["device"], test=True)) for batch
                                       in iter(self.upstream_loader)]
                        TRR = torch.tensor(metric_list)
                        TRR = TRR[~torch.isnan(TRR)]  # Drop nans
                        TRR = torch.mean(TRR.nanmean()).squeeze()

                        # --- Log metrics and push to Weights & Biases ---
                        log_dict["TRR"] = {'UP': TRR.item()}  # Test Retention Rate
                        log_dict["ERR"] = {'HIS': ERR.item()}  # Error Retention Rate
                        log_dict["ES"] = ES  # Edit Success
                        log_dict["train_time"] = edit_time / 60  # Time it takes to make one edit
                        log_dict["edit"] = batch["text"]  # Raw edit input
                        log_dict["edit_label"] = batch["labels"]  # Raw edit label
                        log_dict["n_edits"] = n_edits  # Raw edit label

                        log_dict["ARR"] = ARR  # Accurate Retention Rate


                        LOG.info(f"Number of edits: {n_edits}")
                        for k in log_dict:
                            LOG.info(f"[+edit results+]{k}: {log_dict[k]}")
                        all_UP[n_edits] = log_dict["TRR"]
                        all_HIS[n_edits] = log_dict["ERR"]
                        all_ARR[n_edits] = log_dict["ARR"]

                        all_edit_time[n_edits] = total_edit_time

        with open(f'log.pkl', 'wb') as f:
            pickle.dump(
                {'all_UP': all_UP, 'all_HIS': all_HIS, 'all_ARR': ARR, 'all_edit_time': all_edit_time}, f)
        LOG.info(f"[**Total Edit Time**] {total_edit_time / 60} mins")
