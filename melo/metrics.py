import torch
from utils import *
import logging
LOG = logging.getLogger(__name__)


# DEPRECATED
def sent_success(pre_edit_probs, post_edit_probs, pos_mask, eps=torch.finfo(torch.float32).eps, batch_size=20):
    assert False, "No longer used"
    # content_score = post_edit_probs[pos_mask].prod() ** (1/pos_mask.sum()) / (pre_edit_probs[pos_mask]. + eps)
    post_pos_avg = post_edit_probs[pos_mask].prod() ** (1 / pos_mask.sum())
    pre_pos_avg = pre_edit_probs[pos_mask].prod() ** (1 / pos_mask.sum())
    content_score = post_pos_avg / (pre_pos_avg + eps)
    z_content = min(1., content_score)

    # compute z_sent through a weighting objective
    # normalized_probs = post_edit_probs / (post_edit_probs.sum() + eps)
    # balancing_factor = 0.5 * ((~pos_mask).float().sum() / pos_mask.float().sum() + 1)
    # z_sent_weight = balancing_factor * normalized_probs.dot(pos_mask.float())
    post_neg_avg = post_edit_probs[~pos_mask].prod() ** (1 / (~pos_mask).sum())
    neg_over_pos = post_neg_avg / (eps + post_pos_avg)
    z_sent_weight = 1 / (1 + neg_over_pos)

    # compute z_sent through a ranking objective
    batch_mask = pos_mask.view(-1, batch_size).long()
    sort_idxs = post_edit_probs.view(-1, batch_size).sort(-1, descending=True).indices
    ranked_mask = batch_mask.gather(1, sort_idxs)
    true_mask = batch_mask.sort(-1, descending=True).values
    z_sent_rank = (ranked_mask == true_mask).float().mean()

    # compute the final success scores
    weight_success = (z_content * z_sent_weight) ** 0.5
    rank_success = (z_content * z_sent_rank) ** 0.5

    correct_probs = post_edit_probs[pos_mask].mean()
    wrong_probs = post_edit_probs[~pos_mask].mean()

    return {
        "acc_weight": weight_success,
        "acc_rank": rank_success,
        "rank_score": z_sent_rank,
        "weight_score": z_sent_weight,
        "content_score": content_score,
        "post_edit_probs": post_edit_probs.sum(),
        "pre_edit_probs": pre_edit_probs.sum(),
        "correct_probs": correct_probs,
        "wrong_probs": wrong_probs
    }



# For zsRE and F-NLI
def retain_rate(pre_logits, post_logits, mask=None):
    if pre_logits.shape[-1] == 1:
        pre_logits = pre_logits.squeeze(-1)
    if post_logits.shape[-1] == 1:
        post_logits = post_logits.squeeze(-1)

    assert pre_logits.shape == post_logits.shape
    assert pre_logits.shape[0] == mask.shape[0]

    if pre_logits.dim() == 1:
        # binary classification
        pre_preds = pre_logits > 0
        post_preds = post_logits > 0
        retain = (pre_preds == post_preds).float().mean()
    elif pre_logits.dim() == 3:
        # sequence modeling
        pre_preds = pre_logits.argmax(-1)
        post_preds = post_logits.argmax(-1)
        match = (pre_preds == post_preds) * mask
        retain = (match.sum(-1) == mask.sum(-1)).float().mean()
    else:
        raise NotImplementedError

    return retain.item()


def is_acc_error(model, tokens):
    # Check whether or not the model's prediction for a batch element is correct
    labels = tokens["labels"]
    logits = model(**tokens).logits
    probs = torch.softmax(logits, -1).squeeze()
    argmaxs = torch.argmax(probs, dim=-1).squeeze()
    return labels != argmaxs


def Accuracy(alg, tokens):
    labels = tokens["labels"]
    new_tokens = {f"{k}": v for k, v in tokens.items() if k != "labels"}
    logits = alg.model(**new_tokens).logits
    probs = torch.softmax(logits, -1).squeeze()
    argmaxs = torch.argmax(probs, dim=-1).squeeze()
    return (labels == argmaxs).float().mean()


def is_qa_error(model, tokens):
    preds = model.generate(tokens["input_ids"], max_length=20).squeeze()  # Run model to get its predictions
    labels = tokens["labels"]  # [tokens["labels"] != -100]

    if (len(preds) != len(labels)) or ((preds == labels).sum() != len(preds)):
        return True
    else:
        return False


def PPL(alg, batch):
    input_ids = batch["input_ids"][:, :1024]  # .to(device)
    if "labels" not in batch:
        batch["labels"] = batch["input_ids"][:, :1024].clone()
    else:
        batch["labels"] = batch["labels"][:, :1024].clone()

    with torch.no_grad():
        #outputs = alg.model.model(input_ids=input_ids, labels=target_ids)
        outputs = alg.model(**batch)
        nll = outputs.loss

    ppl = torch.exp(nll)  # .clip(0, 100)
    return ppl



def F1_ACC(alg, batch):
    try:
        preds = alg.generate(batch["input_ids"], max_length=20).squeeze()
        f1 = F1(preds, batch, alg.model_tok)
        acc = ACC(preds, batch, alg.model_tok)
        return f1, acc
    except Exception as e:
        raise e

def F1(preds, batch, tok):
    try:
        f1_list = []
        for p, g in zip(preds,batch["labels"]):
            p = p[p !=  tok.pad_token_id].cpu().squeeze()
            g = g[g != -100].cpu().squeeze()  # -100 might be nonsense
            num_same = len(np.intersect1d(p, g))
            len_pred = len(p)
            len_gold = len(g)
            precision = num_same / len_pred
            recall = 1.0 * num_same / len_gold
            f1 = (2 * precision * recall) / (precision + recall)
            f1_list.append(f1)
    except:
        return 0.

    return sum(f1_list) / len(f1_list)


def ACC(preds, batch, tok):
    decode_preds = tok.batch_decode(preds,skip_special_tokens=True)
    gold_labels = batch['labels']
    gold_labels = gold_labels.masked_fill(gold_labels == -100,tok.pad_token_id)
    decode_labels = tok.batch_decode(gold_labels,skip_special_tokens=True)
    assert len(decode_labels) == len(decode_preds), "Lengths of decode_preds and decode_labels should be the same"
    count = 0.
    for pred,label in zip(decode_preds, decode_labels):
        if pred == label:
            count = count + 1
    return count/len(decode_preds)


import torch
import numpy as np
import scipy
import nltk
import typing
import torch.nn.functional as F

import typing
from itertools import chain
from typing import List, Optional

from transformers import AutoTokenizer
from sklearn.metrics import f1_score
def slice_list(matrix,start_indices,left):
    if isinstance(matrix[0], list):
        if left:
            return [row[start_index-1:-1] for row, start_index in zip(matrix, start_indices)]
        else:
            return [row[start_index:] for row, start_index in zip(matrix, start_indices)]
    else:
        if left:
            return matrix[start_indices[0]-1:-1]
        else:
            return matrix[start_indices[0]:]
        
def test_prediction_acc(model, tok, prompts, targets, device, locality=False, vanilla_generation=False):
    if vanilla_generation:
        if isinstance(prompts, str):
            prompts, targets = [prompts, ], [targets, ]
        results = []
        for prompt, target_new in zip(prompts, targets):
            target_new_tokens = tok.encode(target_new, add_special_tokens=False)
            prompt_tok = tok(
                prompt,
                return_tensors="pt",
            ).to(device)
            gen_token = model.generate(
                input_ids=prompt_tok['input_ids'],
                attention_mask=prompt_tok['attention_mask'],
                max_new_tokens=len(target_new_tokens),
                pad_token_id=tok.eos_token_id,
                do_sample=False,
                use_cache=False,
            )
            if locality:
                results.append(gen_token.detach().cpu().numpy().tolist()[0][-len(target_new_tokens):])
            else:
                results.append(np.mean(np.equal(target_new_tokens, gen_token.detach().cpu().numpy().tolist()[0][-len(target_new_tokens):])))
        return results

    if isinstance(prompts, str):
        prompts,targets = [prompts,], [targets,]

    prompt_target = [prompt + target for prompt, target in zip(prompts,targets)]
    max_prompt_len = max([len(tok.encode(_)) for _ in prompt_target]) + 1
    before_padding_side = tok.padding_side
    tok.padding_side = 'left'
    prompt_target_tok = tok(
        prompt_target,
        padding=True,
        truncation=True,
        max_length=100,
        return_tensors="pt",
    ).to(device)
    prompt_tok = tok(
        prompts,
        padding=True,
        truncation=True,
        max_length=100,
        return_tensors="pt",
    )
    tok.padding_side = before_padding_side
    num_prompt_toks = [int((i != tok.pad_token_id).sum()) for i in prompt_tok['input_ids']]
    num_pad_toks = [int((i == tok.pad_token_id).sum()) for i in prompt_target_tok['input_ids'].cpu()]
    prompt_len = [x+y for x,y in zip(num_pad_toks,num_prompt_toks)]
    with torch.no_grad():
        outputs = model(**prompt_target_tok)
        if type(outputs) is torch.Tensor:
            logits = outputs
        else:
            logits = outputs.logits
        answers = torch.argmax(logits, dim=-1).squeeze().detach().cpu().numpy().tolist()
        labels = prompt_target_tok['input_ids'].squeeze().detach().cpu().numpy().tolist()
        answers = slice_list(answers,prompt_len,left=True)  #由于是预测，得往前平移一个位置截断，left设置为true
        labels = slice_list(labels,prompt_len,left=False)
        if locality:
            return answers if type(answers[0]) is list else [answers,]
        if isinstance(answers[0], list):
            res = []
            for ans,label in zip(answers,labels):
                temp_acc = np.mean(np.equal(ans, label))
                if np.isnan(temp_acc):
                    continue
                res.append(temp_acc)
            return res
        else:
            return [np.mean(np.equal(answers, labels))]

def compute_edit_quality(
    model,
    model_name,
    tok: AutoTokenizer,
    record: typing.Dict,
    device,
    eval_metric: str = 'token_em',
    test_generation = False
) -> typing.Dict:
    """
    Given a rewritten model, computes generalization and specificity metrics for
    the desired rewrite (passed in via the CounterFact dataset record). Returns a
    dictionary containing those metrics.

    :param model: Rewritten model
    :param tok: Tokenizer
    :param record: CounterFact dataset record
    :paran snips: ???
    :param vec: ???
    :return: Dictionary containing rewriting metrics
    """

    # First, unpack rewrite evaluation record.
    target_new, ground_truth = (
        record[x] for x in ["target_new", "ground_truth"]
    )

    rewrite_prompts = record["prompt"]
    rephrase_prompts = record["rephrase_prompt"] if 'rephrase_prompt' in record.keys() else None
    ret = compute_rewrite_or_rephrase_quality(model, model_name, tok,
                                              rewrite_prompts, target_new, device=device, eval_metric=eval_metric)

    ret['locality'] = {}
    ret['portability'] = {}
    if rephrase_prompts is not None:
        ret.update(
            compute_rewrite_or_rephrase_quality(model, model_name, tok,
                                                rephrase_prompts, target_new, device=device, test_rephrase=True, eval_metric=eval_metric)
        )

    if 'locality' in record.keys() and any(record['locality']):
        for locality_key in record['locality'].keys():
            ret['locality'].update(
                compute_locality_quality(model, model_name, tok, locality_key,
                                         record['locality'][locality_key]['prompt'],
                                         record['locality'][locality_key]['ground_truth'], device=device)
            )
    if 'portability' in record.keys() and any(record['portability']):
        for portability_key in record['portability'].keys():
            ret['portability'].update(
                compute_portability_quality(model, model_name, tok, portability_key,
                                            record['portability'][portability_key]['prompt'],
                                            record['portability'][portability_key]['ground_truth'], device=device)
            )
    return ret

def compute_rewrite_or_rephrase_quality(
    model,
    model_name,
    tok: AutoTokenizer,
    prompt: str,
    target_new: str,
    device,
    test_rephrase: bool = False,
    eval_metric: str = 'token_em'
) -> typing.Dict:
    
    if not test_rephrase:
        key = 'rewrite'
    else:
        key = 'rephrase'
    
 

    acc = test_prediction_acc(model, tok, prompt, target_new, device)
    ret = {
        f"{key}_acc": acc
    }
    return ret

def compute_locality_quality(
    model,
    model_name,
    tok: AutoTokenizer,
    locality_key: str,
    prompt: typing.Union[str, List[str]],
    locality_ground_truth: typing.Union[str, List[str]],
    device,
) -> typing.Dict:


    loc_tokens = test_prediction_acc(model, tok, prompt, locality_ground_truth, device, locality=True)

    if type(loc_tokens) is not list:
        loc_tokens = [loc_tokens,]

    ret = {
        f"{locality_key}_output": loc_tokens
    }
    return ret

def compute_portability_quality(
    model,
    model_name,
    tok: AutoTokenizer,
    portability_key: str,
    prompt: typing.Union[str, List[str]],
    ground_truth: typing.Union[str, List[str]],
    device,
) -> typing.Dict:


    portability_correct = test_prediction_acc(model, tok, prompt, ground_truth, device)

    ret = {
        f"{portability_key}_acc": portability_correct
    }
    return ret