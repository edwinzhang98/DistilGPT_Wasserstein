import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW
import deepspeed
import random
import json
import math
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    GenerationConfig)
from transformers import get_constant_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup
from torch.optim.lr_scheduler import CosineAnnealingLR

from lm_datasets import LMTrainDataset
from utils import get_optimizer_params, get_optimizer_params_peft, print_args, initialize
from utils import print_rank, get_rank
from utils import save_rank
from utils import all_gather
from utils import save_parallel
from utils import get_tokenizer, get_model

import string
import json
import os
import argparse
from rouge_score import rouge_scorer
from transformers import AutoTokenizer


default_rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

# adapted the flowing from Squad v1.1 evaluation, without removing the articles.
def normalize_answer(s):
    """Lower text and remove punctuation, and extra whitespace."""

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))


def exact_match(prediction, ground_truth, xlingual=False):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def rouge(prediction, ground_truth, xlingual=False):
    scorer = default_rouge_scorer
    scores = scorer.score(prediction=prediction, target=ground_truth)
    return scores["rougeL"].fmeasure


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths, xlingual=False):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth, xlingual=xlingual)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def compute_metrics(predictions, references, xlingual=False):
    # assert len(predictions) == len(references), f"# of predictions {len(predictions)} doesn't match # of references {len(references)}."
    
    min_length = min(len((predictions)), len(references))
    predictions = predictions[:min_length]
    references = references[:min_length]
    
    em, rougeL = 0, 0
    for pred, gold in zip(predictions, references):
        assert isinstance(gold, list)
        em += metric_max_over_ground_truths(
            exact_match, prediction=pred, ground_truths=gold, xlingual=xlingual
        )
        rougeL += metric_max_over_ground_truths(
            rouge, prediction=pred, ground_truths=gold, xlingual=xlingual
        )
    em = 100.0 * em / len(references)
    rougeL = 100.0 * rougeL / len(references)
    metrics = {"exact_match": em, "rougeL": rougeL}
    metrics = {k: round(v, 4) for k, v in metrics.items()}
    return metrics


def compute_grouped_metrics(predictions, references, groups, xlingual=False):
    assert len(predictions) == len(references) == len(groups)

    examples_by_group = {}
    for pred, gold, group in zip(predictions, references, groups):
        if group not in examples_by_group:
            examples_by_group[group] = []
        examples_by_group[group].append((pred, gold))
    
    results = {}
    for group, group_examples in examples_by_group.items():
        task_predictions, task_references = zip(*group_examples)
        group_metrics = compute_metrics(task_predictions, task_references, xlingual=xlingual)
        for metric, value in group_metrics.items():
            results[f"{metric}_for_{group}"] = value
    return results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prediction_file", required=True,
        help="Jsonl file with each line corresponding to a prediction. " 
             "Each json object should have an `id` and a `prediction` key.")
    parser.add_argument(
        "--reference_file", required=True,
        help="Jsonl file with each line corresponding to a reference. " 
             "Each json object should have an `id` and a `references` key. "
             "`task_id`, `task_category` and `task_track` are optional, which will be used to "
             "compute the per-task performance, per-category performance and the performance for default (english) / xlingual Tracks.")
    parser.add_argument(
        "--output_file",
        help="Jsonl file to write the results to.")
    parser.add_argument(
        "--model_name",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    references = []
    with open(args.reference_file) as fin:
        for line in fin:
            instance = json.loads(line)
            if isinstance(instance["output"], list):
                references.append(instance["output"])
            else:
                references.append([instance["output"]])

    predictions = []
    with open(args.prediction_file) as fin:
        for line in fin:
            prediction = json.loads(line)
            predictions.append(prediction["text"])

    predictions = predictions[:1000]

    references = references[:len(predictions)]

    results = compute_metrics(predictions, references, xlingual=False)

    print(results)

    if args.output_file:
        os.makedirs(args.output_file, exist_ok=True)
        with open(os.path.join(args.output_file, f"{args.model_name}.json"), "w") as fout:
            json.dump(results, fout, indent=2)
            

def evaluate(args, tokenizer, model, dataset: LMTrainDataset, split, epoch, device):
    
    collate_fn = dataset.collate

    #if args.model_parallel:
        #dp_world_size = mpu.get_data_parallel_world_size()
        #dp_rank = mpu.get_data_parallel_rank()
        #dp_group = mpu.get_data_parallel_group()
        #loss_func = mpu.parallel_cross_entropy
    if True:
        dp_world_size = dist.get_world_size()
        dp_rank = dist.get_rank()
        dp_group = None
        loss_func = nn.CrossEntropyLoss()

    print_rank("dp size", dp_world_size)

    generation_config = GenerationConfig(
        do_sample=args.do_sample,
        top_p=args.top_p,
        top_k=args.top_k,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        max_length=args.max_length,
        min_length=None,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
        output_scores=False
    )

    sampler = DistributedSampler(dataset, shuffle=False, drop_last=False, rank=dp_rank, num_replicas=dp_world_size)
    dataloader = DataLoader(
        dataset, sampler=sampler, batch_size=args.eval_batch_size, num_workers=args.num_workers, collate_fn=collate_fn)

    model.eval()
    all_loss = 0.0
    step = 0
    
    all_response_ids = []
    
    with torch.no_grad():
        for it, (model_batch, no_model_batch, gen_data) in enumerate(dataloader):
            # dist.barrier()
            # for rank in range(dist.get_world_size()):
            #     if dist.get_rank() == rank:
            #         print(f"rank: {dist.get_rank()}", model_batch["input_ids"][0][:128])
            #     dist.barrier()
            #print_rank(f"{it}/{len(dataloader)}")
            dataset.move_to_device(model_batch, no_model_batch, gen_data, device)
            logits = model(**model_batch).logits
            if args.model_parallel:
                lm_losses = loss_func(logits.contiguous().float(), no_model_batch["label"]).view(-1)
                loss_mask = no_model_batch["loss_mask"].view(-1)
                loss = (lm_losses * loss_mask).sum(-1) / loss_mask.sum(-1)
            else:
                loss = loss_func(logits.view(-1, logits.shape[-1]), no_model_batch["label"].view(-1))
            
            max_new_tokens = args.max_length - gen_data["input_ids"].size(1)
            
            if args.eval_gen:            
                gen_out = model.generate(
                    **gen_data,
                    generation_config=generation_config,
                    max_new_tokens=max_new_tokens)
                
                full_ids = gen_out.sequences
                
                full_ids = F.pad(
                    full_ids,
                    (0, args.max_length - full_ids.shape[1]),
                    value=tokenizer.pad_token_id,
                )
                
                response_ids = full_ids[:, gen_data["input_ids"].size(1):]
                all_response_ids.append(response_ids)
                    
            dist.all_reduce(loss, dist.ReduceOp.SUM, group=dp_group)
            loss = loss / dp_world_size
            all_loss += loss.item()
            step += 1
    
    if args.eval_gen:
        all_response_ids = torch.cat(all_response_ids, dim=0)
        all_response_ids = all_gather(all_response_ids, dim=1, world_size=dp_world_size, group=dp_group, op="stack")
        all_response_ids = all_response_ids.view(-1, all_response_ids.size(-1))
        
        responses = tokenizer.batch_decode(all_response_ids, skip_special_tokens=True)
    
    if get_rank() == 0:
        if args.eval_gen:
            references = dataset.answers
            responses = responses[:len(references)]
            
            res = compute_metrics(responses, references)
        
            #eval_dir = os.path.join(args.save, "eval", str(epoch))
            #print_rank(eval_dir)
            #os.makedirs(eval_dir, exist_ok=True)
            #with open(os.path.join(eval_dir, "answers.jsonl"), "w") as f:
                #for resp in responses:
                    #f.write(json.dumps({"text": resp}) + "\n")
        else:
            res = {}
    
        avg_loss = all_loss / step
        
        log_str = f"{split} | avg_loss: {avg_loss} | {res}"
        print_rank(log_str)
        #save_rank(log_str, os.path.join(args.save, "log.txt"))
