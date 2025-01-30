from typing import List
from omegaconf import OmegaConf
import torch
import copy
import transformers
import logging
import os

from torch.nn import Parameter

from utils import *

from peft import (
    PeftModel,
    prepare_model_for_int8_training,
    MeloConfig,
    get_peft_model,
    get_peft_model_state_dict,
)
from peft.tuners.melo import LoraLayer, GraceLayer
from hooks import lora_backward_hook
# from models import BertClassifier
LOG = logging.getLogger(__name__)
def translate_tokens(tokens, from_tok, to_tok):
    tokens = tokens.masked_fill(tokens == -100, from_tok.pad_token_id)
    text = from_tok.batch_decode(tokens, skip_special_tokens=True)
    return to_tok(text, return_tensors="pt")["input_ids"].to(tokens.device)

class LORA(torch.nn.Module):
    def __init__(self, model, config, model_tok,scale=None):
        super(LORA, self).__init__()
        self.config = config

        '''Apply_lora
        '''
        r_num = config.grace.num_block * config.grace.num_rank_per_block
        self.lora_config = MeloConfig(
            r = r_num,
            lora_alpha = r_num,
            target_modules= list(config.model.target_modules),
            lora_dropout = config.lora.lora_dropout,
            task_type = config.lora_task_type,
            fan_in_fan_out= config.model.fan_in_fan_out,
            grace_layer = config.model.grace_layer,
            grace_config= OmegaConf.to_object(config.grace)
        )
        self.log_dict = {}

        '''Load
        '''
        # self.original_model = model
        # self.model = model

        if not config.check_dir:
            self.model = get_peft_model(model, self.lora_config)
        else:
            save_path = os.path.join(config.base_dir, "checkpoint", config.check_dir)
            self.load_from_checkpoint(save_path)

        self.lora_list = self.named_lora_modules()
        self.grace_layer = self.named_grace_layer()
        # self.register_lora_backward_hooks(lora_backward_hook)

        '''Load Tokenizer
        '''
        self.model_tok = model_tok
        self.classifier_tok = transformers.AutoTokenizer.from_pretrained(config.lora.cls_name)

        '''Parameters to be optimized
        '''
        self.opt_params = self.optim_parameters()
        pass


    def optim_parameters(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad==True and 'lora' not in name:
                param.requires_grad = False
        lora_params = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        return lora_params




    #TODO
    def load_from_checkpoint(self, save_path):
        print(save_path)


    def save_classifier_weights(self,cls_dir):
        if not os.path.exists(cls_dir):
            os.makedirs(cls_dir,exist_ok=True)
        torch.save(self.classifier.state_dict(),f"{cls_dir}/classifier.pt")
    def save_lora_weights(self,lora_dir):
        self.model.save_pretrained(lora_dir+"/lora_checkpoint")


    def reset_lora(self):
        for key in self.lora_list:
            self.model.get_submodule(key).reset_lora_parameters('default')

    def named_lora_modules(self):
        module_list = [key for key,_ in self.model.named_modules()]
        lora_list = []
        for key in module_list:
            if isinstance(self.model.get_submodule(key),LoraLayer):
                lora_list.append(key)
        return lora_list

    def named_grace_layer(self) -> str:
        module_list = [key for key, _ in self.model.named_modules()]
        grace_list = []
        for key in module_list:
            if isinstance(self.model.get_submodule(key), GraceLayer):
                grace_list.append(key)
        assert len(grace_list) == 1, "At Most One Grace Layer"
        return grace_list[0]

    def register_lora_backward_hooks(self,backward_hook_fn):
        for key in self.lora_list:
            self.model.get_submodule(key).register_backward_hook(backward_hook_fn)


    def disable_melo(self):
        self.model.base_model.disable_adapter_layers()
        self.model.base_model.disable_grace_layer()

    def enable_melo(self):
        self.model.base_model.enable_adapter_layers()
        self.model.base_model.enable_grace_layer()


    def edit(self, tokens):

        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]
        labels = tokens["labels"]
        labels[labels == 128000] = -100  # 将128000替换为-100
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        
        optimizer = torch.optim.Adam(self.optim_parameters(), self.config.grace.edit_lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=20,gamma=0.5)
        # --- pass edit label, training mode, and key_id into GRACE ---
        setattr(self.model.get_submodule(self.grace_layer), "training", True)
        setattr(self.model.get_submodule(self.grace_layer), "edit_label", tokens["labels"])
        

        self.losses = []
        for i in range(self.config.grace.num_iter):
            # --- insert iteration into each layer (only initiate keys on first iteration) ---
            setattr(self.model.get_submodule(self.grace_layer), "batch_iter", i)

            # --- pass tokens through model (including through the GRACE layer) ---
            outputs = self.model.model(**inputs)
            loss=multiclass_log_probs(self.config,outputs.logits,labels,shift=True)['nll']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            self.losses.append(loss.detach().cpu().numpy())
            LOG.info(f'batch loss in iter {i}: {loss.detach().cpu().numpy()}')
        self.loss = loss # Log final loss

        setattr(self.model.get_submodule(self.grace_layer), "training", False)




    def generate(self, *args, **kwargs):
        return self.model.model.generate(*args, **kwargs)

    def get_VecDB_info(self):
        VecDB_logdict = {}
        VecDB_logdict["num_cluster"] = len(getattr(self.model.get_submodule(self.grace_layer), "VecDB"))
        VecDB_logdict["conflict_num"] = getattr(self.model.get_submodule(self.grace_layer), "VecDB").conflict_num
        VecDB_logdict["forget_keys"] = len(getattr(self.model.get_submodule(self.grace_layer), "VecDB").forget_keys)
        return VecDB_logdict

def masked_mean(values, mask):
    assert mask.dtype == torch.bool
    assert values.shape == mask.shape
    return (values * mask.float()).sum() / mask.sum().float()
def multiclass_log_probs(config, pred, targ, shift=False, eps=torch.finfo(torch.float32).eps, exact_match=False, **kwargs):
    NULL_TOKEN = 0  # a placeholder used for masked target locations

    pred = pred.clone()
    targ = targ.clone()
    if shift and pred.dim() == 3:  # Dealing with sequences
        pred = pred[:, :-1]  # Remove last prediction in sequence
        if "inner_sent" in kwargs or "personality" in kwargs or "multimodal" in kwargs:
            targ = targ[:, 1:]
        else:
            pred = pred[:, -targ.size(1):]
        # targ = targ[:, 1:]  # Shift to align predictions and targets

    mask = targ != -100
    targ[~mask] = NULL_TOKEN  # Can be any valid token, since we'll throw them out
    unmasked_log_probs = pred.log_softmax(-1).gather(-1, targ.unsqueeze(-1)).squeeze(-1)
    
    # debug
    # print(pred.shape, targ.shape)
    # if pred.size(1) > targ.size(1):
    #     pred = pred[:, :targ.size(1)]

    if exact_match:
        pred_ids = pred.argmax(-1).masked_fill(~mask, NULL_TOKEN)
        correct = pred_ids == targ
        if pred.dim() == 3:
            correct = (pred_ids == targ).all(-1)  # We aim for an exact match across the entire sequence
        acc = correct.float().mean()
    else:
        pred_ids = pred.argmax(-1).masked_fill(~mask, NULL_TOKEN)
        correct = pred_ids == targ
        correct = correct & mask
        num_non_padding = mask.sum().float().item()
        acc = correct.sum() / num_non_padding
    
    if "inner_sent" in kwargs or "inner_per" in kwargs:
        same_sent_mask = kwargs["same_mask"]
        good_mask = mask * same_sent_mask.unsqueeze(-1)
        bad_mask = mask * (~same_sent_mask.unsqueeze(-1))

        good_log_prob = masked_mean(unmasked_log_probs, good_mask)
        bad_log_prob = masked_mean((1 - unmasked_log_probs.exp() + eps).log(), bad_mask)

        n_tokens = good_mask.float().sum()
        log_prob = good_log_prob
        prob = log_prob.exp()

        if kwargs["unlikelihood"]:
            nll = -good_log_prob - bad_log_prob
        else:
            nll = -good_log_prob
    else:
        n_tokens = mask.float().sum()
        log_prob = (unmasked_log_probs * mask.float()).sum() / n_tokens
        prob = (unmasked_log_probs.exp() * mask.float()).sum() / n_tokens
        
        nll = -log_prob
    return {
        "acc": acc,
        "log_prob": log_prob,
        "prob": prob,
        "n_tokens": n_tokens,
        "nll": nll,
    }







if __name__ == '__main__':
    pass


















