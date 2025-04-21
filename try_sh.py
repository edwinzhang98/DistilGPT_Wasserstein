from arguments import get_args
from lm_datasets import LMTrainDataset
from transformers import GPT2Config, GPT2LMHeadModel
from utils import get_tokenizer, get_model, get_teacher_model
from critic import critic
#from evaluate import evaluate
import evaluate
import random
import torch
import os
from torch.utils.data import Subset
import torch.distributed as dist
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
from datasets import Dataset, load_from_disk
from torch.nn.parallel import DistributedDataParallel as DDP
from train_related import save_student_model, example_gen, top_p_sampling, get_student_model, \
    get_gpt2dolly, get_distillgpt2, getgpt2, calculate_perplexity, preprocess_function, \
    evaluate_model_with_rouge, get_distillgpt2_with_reset, get_trained
from cnn_dataset import CNNDMDistillationDataset, OpenWebTextDataset
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.nn as nn
# Add necessary imports for plotting
import matplotlib
matplotlib.use('Agg') # Use Agg backend for non-GUI environments
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
# import deepspeed # Remove deepspeed import

class cfg:
    def __init__(self, distill):
        self.distill_step = distill

        

def get_critics_and_optimizers(n, dim, device, lr=1e-4):
    # n := number of layers need distillation
    # dim := embedding dimension 
    # lr := learning rate
    critics = []
    optimizers = []
    schedulers = []
    
    for i in range(n):
        # 定义 Critic
        new_critic = critic(dim)
        new_critic = new_critic.to(device)
        critics.append(new_critic)
        
        # 定义对应的 Optimizer
        optimizer = optim.RMSprop(new_critic.parameters(), lr=lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=1000)
        optimizers.append(optimizer)
        schedulers.append(scheduler)
        
    print(f"There are {n} critics and optimizers")
        
    return critics, optimizers, schedulers


import os

import torch
import torch.nn.functional as F

def compute_topk_kl_loss(teacher_logits, student_logits, temperature=2, top_k=10):
    """
    只对教师模型前 top_k 个高概率 token 计算KL散度，
    并且在 top_k 内部重新归一化分布再计算 KL。
    """

    # 1. 计算教师、学生在整个词表上的 softmax 分布
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)   # [batch_size, seq_len, vocab_size]
    student_probs = F.softmax(student_logits / temperature, dim=-1)

    # 2. 获取教师模型每个位置概率最高的前 top_k 个 token
    #    以及相应的概率分布
    #    shape: [batch_size, seq_len, top_k], [batch_size, seq_len, top_k]
    teacher_topk_probs, teacher_topk_indices = teacher_probs.topk(top_k, dim=-1)

    # 3. 对教师 top_k 概率做归一化，使这 top_k 个值之和为 1
    #    这样就获得了教师在 top_k 子集上的分布
    teacher_topk_probs = teacher_topk_probs / (teacher_topk_probs.sum(dim=-1, keepdim=True) + 1e-12)

    # 4. 学生同样只取对应的 top_k 索引位置，再做归一化
    #    shape: [batch_size, seq_len, top_k]
    student_topk_probs = torch.gather(student_probs, dim=-1, index=teacher_topk_indices)
    student_topk_probs = student_topk_probs / (student_topk_probs.sum(dim=-1, keepdim=True) + 1e-12)

    # 5. 计算 KL(Teacher||Student)，只在 top_k 的分布上计算
    #    KL(P||Q) = sum(P * log(P/Q))
    #    注意需要乘以 temperature^2
    kl = teacher_topk_probs * (
        torch.log(teacher_topk_probs + 1e-12) - torch.log(student_topk_probs + 1e-12)
    )
    # Revert loss type casting if any
    kl_loss = kl.sum(dim=-1).mean() * (temperature**2)

    return kl_loss


def get_topk_probs(teacher_logits, student_logits, temperature=1.0, top_k=50):
    """
    从教师模型的 logits 中获取概率最高的 top_k token，
    然后在对应 token 上构造学生的分布（通过 gather）。
    返回:
      teacher_topk_probs, teacher_topk_indices, student_topk_probs
    形状: [batch_size, seq_len, top_k]
    """
    # 1. 先计算 softmax 概率
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)  # [B, L, V]
    student_probs = F.softmax(student_logits / temperature, dim=-1)

    # 2. 获取教师最高的 top_k 概率及其索引
    #    shape: [B, L, top_k]
    teacher_topk_probs, teacher_topk_indices = teacher_probs.topk(top_k, dim=-1)  
    
    # 3. 对应地在学生分布中取相同的 token 索引
    #    shape: [B, L, top_k]
    student_topk_probs = torch.gather(student_probs, dim=-1, index=teacher_topk_indices)

    # 4. 在这 top_k 内做归一化，让它们之和为 1
    #    避免除 0，通常加个极小量
    teacher_topk_probs = teacher_topk_probs / (teacher_topk_probs.sum(dim=-1, keepdim=True) + 1e-12)
    student_topk_probs = student_topk_probs / (student_topk_probs.sum(dim=-1, keepdim=True) + 1e-12)

    return teacher_topk_probs, teacher_topk_indices, student_topk_probs

def wasserstein_distance_1d(
    teacher_topk_probs: torch.Tensor,   # [B, L, K]
    teacher_topk_indices: torch.Tensor, # [B, L, K]
    student_topk_probs: torch.Tensor    # [B, L, K]
) -> torch.Tensor:
    """
    基于「教师 top_k token 及其 ID」在一维数轴上计算离散 1D Wasserstein 距离。

    1) 首先对 token ID (teacher_topk_indices) 在最后一维排序，
    2) 分别累加得到教师CDF、学生CDF，
    3) 利用梯形法对 |CDF_t(i) - CDF_s(i)| 乘以相邻 token ID 差做积分近似。

    返回标量 (tensor)，表示在 [B, L] 上的平均 Wasserstein 距离。
    """

    # 1. 对最后一维 (K) 做排序，得到升序 token_id 以及对应的排序下标
    #    shape: [B, L, K] (升序)
    sorted_ids, sorted_idx = teacher_topk_indices.sort(dim=-1)

    # 2. 根据 sorted_idx 重新排列 teacher 和 student 的概率
    #    shape: [B, L, K]
    teacher_probs_sorted = torch.gather(teacher_topk_probs, -1, sorted_idx)
    student_probs_sorted = torch.gather(student_topk_probs, -1, sorted_idx)

    # 3. 分别计算 CDF: 即 cumsum
    #    shape: [B, L, K]
    teacher_cdf = teacher_probs_sorted.cumsum(dim=-1)
    student_cdf = student_probs_sorted.cumsum(dim=-1)

    # 4. 准备做梯形法积分:
    #    W1 ~= sum over j in [0..K-2] of 0.5 * (|CDF_t(j)| + |CDF_t(j+1)|) * ( x_{j+1} - x_j )
    #    但这里 x_j = sorted_ids[..., j]，CDF_t(j) = teacher_cdf[..., j]
    #    由于是概率分布，CDF >= 0，无需再取绝对值；但 teacher_cdf - student_cdf 要做绝对值.

    #    4.1 先取相邻 token id 的差
    #        shape: [B, L, K-1]
    xj    = sorted_ids[..., :-1].float()
    xj1   = sorted_ids[..., 1:].float()
    delta = (xj1 - xj)  # 距离

    #    4.2 取相邻 CDF 的差，用梯形法
    cdf_diff       = (teacher_cdf - student_cdf).abs()      # [B, L, K]
    cdf_diff_left  = cdf_diff[..., :-1]                     # [B, L, K-1]
    cdf_diff_right = cdf_diff[..., 1:]                      # [B, L, K-1]

    #    4.3 梯形法：0.5 * (f_left + f_right) * delta_x
    #        shape: [B, L, K-1]
    trapezoid_area = 0.5 * (cdf_diff_left + cdf_diff_right) * delta

    # 5. 对最后一维 (K-1) 求和，得到每个 [B, L] 的 1D-Wasserstein 距离
    wdist = trapezoid_area.sum(dim=-1)  # [B, L]

    # 6. 在 batch 维度和 seq_len 维度上做平均，以得到最终标量 loss
    #    当然，你也可以只在 seq_len 上累加，然后在 batch 上取平均，看需求而定
    wdist_mean = wdist.mean()

    return wdist_mean




def token_level_distill(args, model_data, student_model, teacher_model, critics, critic_optimizer, student_optimizer, student_layer, teacher_layer, generated_ids, attention_mask):
    for _ in range(args.max_input_len - model_data["input_ids"].shape[1]):

        # 前向传播：教师和学生模型
        with torch.no_grad():
            teacher_output = teacher_model(input_ids=generated_ids, attention_mask=attention_mask, output_hidden_states=True)
        student_output = student_model(input_ids=generated_ids, attention_mask=attention_mask, output_hidden_states=True)

        #使用教师logits
        teacher_logits = teacher_output.logits
        next_token_logits = teacher_logits[:,-1,:]
        # 学生模型的 logits
        student_logits = student_output.logits
        student_next_token_logits = student_logits[:, -1, :]
                        
        # 计算 KL loss
        teacher_probs = F.softmax(next_token_logits, dim=-1)
        student_probs = F.log_softmax(student_next_token_logits, dim=-1)
        kl_loss = F.kl_div(student_probs, teacher_probs, reduction='batchmean')

        # Top-p + Temperature
        next_token = top_p_sampling(next_token_logits, top_p=0.9, temperature=0.8)
        # 将生成的 token 添加到输入序列
        next_token = next_token.unsqueeze(1)
        generated_ids = torch.cat([generated_ids, next_token], dim=1)
        #with torch.no_grad():
            # 更新 attention_mask，确保新增的 token 被关注
        new_attention_mask = torch.ones((attention_mask.size(0), 1), device=attention_mask.device)
        attention_mask = torch.cat([attention_mask, new_attention_mask], dim=1)

        # Critic 蒸馏步骤
        critic_loss, student_loss = critic_step(
                                cur_stu_ly=student_layer,
                                cur_tea_ly=teacher_layer,
                                student_model=student_model,
                                student_output=student_output,
                                target=teacher_output,
                                critics=critics,
                                critic_optimizer=critic_optimizer,
                                student_optimizer=student_optimizer,  # 学生模型的 optimizer 放在最后
                                arg=args
                                )
        # 将 KL loss 加入到学生模型的 loss 中
        total_loss = student_loss +  kl_loss
        # 反向传播和优化学生模型
        student_optimizer.zero_grad()
        total_loss.backward()
        student_optimizer.step()
    return total_loss, student_loss, critic_loss, kl_loss
        
        
def distill_step(epoch, step, args, student_model, student_optimizer, teacher_model, critic_list, optimizer_list, input_ids, attention_mask, labels):
    # 前向传播：教师和学生模型
    with torch.no_grad():
        teacher_output = teacher_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, labels=labels)
    student_output = student_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, labels=labels)
    teacher_states = teacher_output.hidden_states
    teacher_states = teacher_output.hidden_states[1:] 
    teacher_states = [teacher_states[(i+1) * 2 -1 ] for i in range(12 // 2)]
    student_states = student_output.hidden_states
    student_states = student_output.hidden_states[1:]
    
    # 使用教师 logits，添加 temperature
    
    temperature = 2
    teacher_logits = teacher_output.logits 
    student_logits = student_output.logits

    # 计算 KL loss
    loss_fct = nn.KLDivLoss(reduction="batchmean")

    kl_loss = loss_fct(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1),  # 修正这里
    ) * (temperature ** 2)
    
    #kl_loss = compute_topk_kl_loss(teacher_logits, student_logits, temperature=2, top_k=200)
    
    """teacher_topk_probs, teacher_topk_indices, student_topk_probs = get_topk_probs(
        teacher_logits, student_logits, temperature=1.0, top_k=500
    )

    kl_loss = wasserstein_distance_1d(
        teacher_topk_probs, 
        teacher_topk_indices, 
        student_topk_probs
    )"""

    
    # MSE LOSS
    # mse = F.mse_loss(torch.stack(student_states, dim=1), torch.stack(teacher_states, dim=1))

    
    # Critic 蒸馏步骤
    critic_loss_container = []
    total_stu_loss = torch.tensor(0.0, device=input_ids.device)
    for student_emb, teacher_emb, critic, optimizer_critic in zip(student_states, teacher_states, critic_list, optimizer_list):
        (critic_loss_avg, student_loss_avg), student_loss = critic_step(student_emb, teacher_emb, critic, optimizer_critic, args)
        total_stu_loss = total_stu_loss + student_loss
        critic_loss_container.append(critic_loss_avg)

        
    # 原始损失（保留计算图）
    original_total_stu_loss = total_stu_loss
    #original_mse = mse
    original_kl_loss = kl_loss
    original_hard_loss = student_output.loss

    # 同步损失（仅用于监控）
    total_stu_loss_sync = total_stu_loss.detach().clone()
    #mse_sync = mse.detach().clone()
    kl_loss_sync = kl_loss.detach().clone()
    hard_loss_sync = student_output.loss.detach().clone() if student_output.loss is not None else torch.tensor(0.0) # Handle None case

    # 执行 all_reduce 和平均（仅同步用于监控的损失）
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(total_stu_loss_sync, op=torch.distributed.ReduceOp.SUM)
        #torch.distributed.all_reduce(mse_sync, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(kl_loss_sync, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(hard_loss_sync, op=torch.distributed.ReduceOp.SUM)
        
        world_size = torch.distributed.get_world_size()
        total_stu_loss_sync = total_stu_loss_sync / world_size
        #mse_sync = mse_sync / world_size
        kl_loss_sync = kl_loss_sync / world_size
        hard_loss_sync = hard_loss_sync / world_size
            

    total_loss = 0.3*original_total_stu_loss + 0.4*original_kl_loss + 0.3*original_hard_loss #+ 0.1*original_mse
    # Remove loss type casting if any
    # total_loss = total_loss.float()

    # Revert to manual gradient accumulation and backward
    if args.gradient_accumulation_steps > 0:
        total_loss = total_loss / args.gradient_accumulation_steps
    total_loss.backward()
    
    # Revert to manual optimizer step and zero_grad based on accumulation
    if (step + 1) % args.gradient_accumulation_steps == 0:
        student_optimizer.step()
        student_optimizer.zero_grad()
    # student_model.backward(total_loss) # Remove DeepSpeed backward
    # student_model.step() # Remove DeepSpeed step

    return -1, total_stu_loss_sync, critic_loss_container, kl_loss_sync

def kl_MSE_step(args, student_model, student_optimizer, teacher_model, critic_list, optimizer_list, input_ids, attention_mask, labels):
    # 前向传播：教师和学生模型
    with torch.no_grad():
        teacher_output = teacher_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, labels=labels)
    student_output = student_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, labels=labels)
    teacher_states = teacher_output.hidden_states
    teacher_states = teacher_output.hidden_states[1:] 
    teacher_states = [teacher_states[(i+1) * 2 - 1] for i in range(12 // 2)]
    student_states = student_output.hidden_states
    student_states = student_output.hidden_states[1:]
    
    # 使用教师 logits，添加 temperature
    temperature = 1
    teacher_logits = teacher_output.logits / temperature
    student_logits = student_output.logits / temperature

    # 计算 KL loss
    teacher_probs = F.softmax(teacher_logits, dim=-1)
    student_probs = F.log_softmax(student_logits, dim=-1)
    kl_loss = F.kl_div(student_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)
    
    # MSE LOSS
    mse = 0
    for student_emb, teacher_emb in zip(student_states, teacher_states):
        mse += F.mse_loss(student_emb, teacher_emb)
        
    # hard label loss
    hard_loss = student_output.loss
    
    total_loss = 0.3*mse + 0.3*kl_loss + 0.4*hard_loss
        
    # 反向传播和优化学生模型
    student_optimizer.zero_grad()
    total_loss.backward()
    student_optimizer.step()

    return hard_loss, None, mse, kl_loss



def gradient_penalty(critic, real_data, fake_data):
    alpha = torch.rand(real_data.size(0), 1, 1).to(real_data.device)
    interpolated = alpha * real_data + (1 - alpha) * fake_data
    interpolated.requires_grad_(True)
    score_interpolated = critic(interpolated)
    gradients = torch.autograd.grad(
        outputs=score_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(score_interpolated),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_norm = gradients.norm(2, dim=1)
    penalty = ((gradient_norm - 1) ** 2).mean()
    return penalty



def critic_step(student_emb, teacher_emb, critic, critic_optimizer, arg):        
    # 开始训练Critic
    for _ in range(arg.critic_time):
        teacher_score = critic(teacher_emb)
        student_output_detached = student_emb.detach()
        student_score_critic = critic(student_output_detached)
        # 判别器的损失：最大化教师评分，最小化学生评分
        critic_loss = -(torch.mean(teacher_score) - torch.mean(student_score_critic))
        gp = gradient_penalty(critic, teacher_emb, student_output_detached)
        critic_loss += arg.lambda_gp * gp
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()
    # 重新计算 student_score，这次不分离计算图
    student_score = critic(student_emb)

    # 学生模型的损失：最小化学生评分
    student_loss = -torch.mean(student_score)
    
       # 同步做日志用的损失，别破坏原图
    critic_loss_for_logging = critic_loss.detach().clone()
    student_loss_for_logging = student_loss.detach().clone()

    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(critic_loss_for_logging, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(student_loss_for_logging, op=torch.distributed.ReduceOp.SUM)
        world_size = torch.distributed.get_world_size()
        critic_loss_avg = critic_loss_for_logging.item() / world_size
        student_loss_avg = student_loss_for_logging.item() / world_size
    else:
        critic_loss_avg = critic_loss_for_logging.item()
        student_loss_avg = student_loss_for_logging.item()

    # 最后返回两份：一个给外部backward用，一个仅仅是日志数值
    return (critic_loss_avg, student_loss_avg), student_loss
    


def train_step(CFG, schedulars, student_scheduler, test_dataset, data_loader, student_model, teacher_model, tokenizer, critics, optimizers, student_optimizer, num_epochs, device, args):
    # Critic -> compute Wasserstein Distance
    # 设置训练模式
    teacher_model.eval()
    student_model.train()
    for each_critic in critics:
        each_critic.train()    
        
    # 得到需要蒸馏的总层数
    n_layers = student_model.module.config.n_layer
    n_layers_teacher = teacher_model.module.config.n_layer
    
    # 开始训练：逐层进行蒸馏
        
    for epoch in range(num_epochs):              
        data_loader.sampler.set_epoch(epoch)
        for step, batch in enumerate(data_loader):
            # --- Print dtype info once per epoch on rank 0 ---
            if step == 0 and dist.get_rank() == 0:
                 print(f"--- Input Data Check (Epoch {epoch}, Step 0) ---")
                 try:
                     # Assuming batch is a dictionary and has 'input_ids'
                     input_ids_dtype = batch["input_ids"].dtype
                     print(f"Input batch ('input_ids') dtype: {input_ids_dtype}")
                 except Exception as e:
                     print(f"Could not check input batch dtype: {e}")
                 print("------------------------------------------")
            # --- End dtype info ---

            # 初始输入 (复制一份作为生成用，避免修改原数据)
            # 数据移至 GPU，遍历数据批次
            #model_data, no_model_data, gen_data = batch
            model_data = batch
            input_ids = model_data["input_ids"].to(device)
            attention_mask = model_data["attention_mask"].to(device)
            labels = model_data["labels"].to(device)
                
            # 自回归生成/不自回归
            time1 = time.time()
            if CFG.distill_step == "kl_MSE_step":
                hard_loss, student_loss, critic_loss_container, kl_loss = kl_MSE_step(args, student_model, student_optimizer, teacher_model, critics, optimizers, input_ids, attention_mask, labels)
            else:
                mse, student_loss, critic_loss_container, kl_loss = distill_step(epoch, step, args, student_model, student_optimizer, teacher_model, critics, optimizers, input_ids, attention_mask, labels)
            # 每 20 步打印一次损失
            if dist.get_rank() == 0 and step % 50 == 0:
                time2 = time.time()
                epsilon = time2 - time1
                if student_loss is None:
                    print(f" Epoch [{epoch}/{num_epochs}] | Step [{step}/{len(data_loader)}] | MSE loss: {critic_loss_container:.4f} |Kl loss: {kl_loss:.4f}| hard loss: {hard_loss.item():.4f}| Time: {epsilon:.4f}")
                else:
                    # Revert step counter if needed (use plain step)
                    current_step = step
                    print(f" Epoch [{epoch}/{num_epochs}] | Step [{current_step}/{len(data_loader)}] | Student Loss: {student_loss:.3f} | C1L: {critic_loss_container[0]:.3f}, C2L: {critic_loss_container[1]:.3f}, C3L: {critic_loss_container[2]:.3f}, C4L: {critic_loss_container[3]:.3f}, C5L: {critic_loss_container[4]:.3f}, C6L: {critic_loss_container[5]:.3f}, | Kl loss: {kl_loss:.4f}| Time: {epsilon:.4f}")
                         
            # 每 400 步生成一次示例
            if step % 400 == 0:
                #t_SNE(student_model, teacher_model, tokenizer, device, step, CFG)
                ppl = calculate_perplexity(student_model, tokenizer, test_dataset, device)
                if dist.get_rank() == 0:
                    print(f"Student Perplexity on validation set: {ppl}")
                # 25.1
            
            if step % 5000 == 0:  
                #print("calculating ppl...")
                #ppl = calculate_perplexity(student_model, tokenizer, test_dataset, device)
                #print(f"Perplexity on validation set: {ppl}")
                dist.barrier()
                if dist.get_rank() == 0:
                    # Revert to original saving method (assuming save_student_model was used)
                    # You might need to uncomment/adjust this based on your original code
                    student_save_path = os.path.join(args.base_path if args.base_path else ".", "output", "distillated_student_KL")
                    os.makedirs(student_save_path, exist_ok=True) #确保目录存在
                    name = 'well-trained'+str(epoch) + str(step) + str(CFG.distill_step)
                    save_student_model(student_model, student_save_path, cur_ly=name)
                    # save_tag = f"epoch{epoch}_step{current_step}_{CFG.distill_step}" # Use double quotes for f-string
                    # student_model.save_checkpoint(args.save, save_tag) # Remove DeepSpeed save_checkpoint
                dist.barrier()
                
            
        student_scheduler.step()
        for each_scheduler in schedulars:
            each_scheduler.step() 
        if dist.get_rank() == 0:
            dist.barrier()
            ppl = calculate_perplexity(student_model, tokenizer, test_dataset, device)
            dist.barrier()
            print(f"Perplexity on validation set: {ppl}")
                        
            # Revert saving method at the end of epoch
            if dist.get_rank() == 0:
            student_save_path = os.path.join(args.base_path if args.base_path else ".", "output", "distillated_student_KL")
            os.makedirs(student_save_path, exist_ok=True) #确保目录存在
            name = 'well-trained'+str(epoch) + str(CFG.distill_step)
            save_student_model(student_model, student_save_path, cur_ly=name)
                 # save_tag = f"epoch{epoch}_end_{CFG.distill_step}"
                 # student_model.save_checkpoint(args.save, save_tag) # Remove DeepSpeed save_checkpoint

# --- Add t_SNE function definition ---
def t_SNE(student, teacher, tokenizer, device, save_path):
    # Prepare example text
    text = "Wasserstein Distillation is a method for training smaller models using larger models. It works by matching the hidden states of the two models."
    
    # Tokenize the input text and move to device
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Ensure models are in eval mode
    student.eval()
    teacher.eval()
    
    # Forward pass with explicit config for hidden states
    with torch.no_grad():
        # Use .module if models are wrapped in DDP
        student_module = student.module if hasattr(student, 'module') else student
        teacher_module = teacher.module if hasattr(teacher, 'module') else teacher
        
        outputs_gpt2 = teacher_module(
            **inputs,
            output_hidden_states=True,
            return_dict=True
        )
        outputs_distilgpt2 = student_module(
            **inputs,
            output_hidden_states=True,
            return_dict=True
        )
    
    # Verify hidden states are available
    if outputs_gpt2.hidden_states is None or outputs_distilgpt2.hidden_states is None:
        print("Warning: Hidden states not returned by the models. Check model configurations. Skipping t-SNE plot.")
        return
        # raise ValueError("Hidden states not returned by the models. Check model configurations.")
    
    # Get tokens for visualization
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    
    # Define layer pairs to align (adjust if student/teacher layers differ significantly)
    # Assuming student has 6 layers and teacher has 12
    student_layers = student_module.config.n_layer
    teacher_layers = teacher_module.config.n_layer

    # Simple pairing strategy (adjust if needed)
    layer_pairs = []
    if student_layers == 6 and teacher_layers == 12:
         # Standard DistilGPT2 pairing
        layer_pairs = [(1, 2), (2, 4), (3, 6), (4, 8), (5, 10), (6, 12)]
    else:
        # Fallback: pair layers proportionally if possible, or just use first/last
        if student_layers > 0 and teacher_layers > 0:
             step = teacher_layers // student_layers
             layer_pairs = [(s_idx + 1, min(teacher_layers, (s_idx + 1) * step)) for s_idx in range(student_layers)]
        else: # Cannot determine pairs
            print("Warning: Cannot determine layer pairs for t-SNE due to layer configuration. Skipping plot.")
            return
        print(f"Using layer pairs for t-SNE: {layer_pairs}")


    # Create subplots
    # Adjust subplot grid based on the number of pairs
    n_pairs = len(layer_pairs)
    ncols = 3
    nrows = (n_pairs + ncols - 1) // ncols # Calculate rows needed
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 6 * nrows))
    axes = axes.flatten() # Flatten to easily index

    # Convert hidden states to list if they're tuples
    hidden_states_gpt2 = list(outputs_gpt2.hidden_states)
    hidden_states_distilgpt2 = list(outputs_distilgpt2.hidden_states)

    plot_idx = 0 # Keep track of which subplot to use
    for distil_layer_idx, gpt2_layer_idx in layer_pairs:
        # Ensure layer indices are valid (0 is embedding layer)
        if distil_layer_idx < 0 or distil_layer_idx >= len(hidden_states_distilgpt2) or \
           gpt2_layer_idx < 0 or gpt2_layer_idx >= len(hidden_states_gpt2):
             print(f"Warning: Invalid layer index in pair ({distil_layer_idx}, {gpt2_layer_idx}). Skipping this pair.")
             continue
             
        try:
            # Get layer outputs and move to CPU for t-SNE
            distil_layer_output = hidden_states_distilgpt2[distil_layer_idx][0].cpu()
            gpt2_layer_output = hidden_states_gpt2[gpt2_layer_idx][0].cpu()
            
            # Concatenate embeddings
            combined_embeddings = torch.cat([distil_layer_output, gpt2_layer_output], dim=0).numpy()
            
            # Calculate safe perplexity value
            n_samples = combined_embeddings.shape[0]
            perplexity = min(30, max(5, n_samples - 1))  # Adjust perplexity range if needed
            if n_samples <= perplexity:
                 perplexity = max(1, n_samples - 1) # Ensure perplexity < n_samples
            if n_samples <= 1: # Cannot run t-SNE with 1 or 0 samples
                print(f"Warning: Not enough samples ({n_samples}) for t-SNE for layer pair ({distil_layer_idx}, {gpt2_layer_idx}). Skipping.")
                continue

            # Perform t-SNE
            tsne = TSNE(
                n_components=2,
                random_state=42,
                init='pca',
                learning_rate='auto',
                perplexity=perplexity,
                n_iter=300 # Reduce iterations for speed if needed
            )
            tsne_results = tsne.fit_transform(combined_embeddings)
            
            # Split results for both models
            n_tokens = distil_layer_output.shape[0]
            distil_points = tsne_results[:n_tokens]
            gpt2_points = tsne_results[n_tokens:]
            
            # Plot points
            ax = axes[plot_idx]
            ax.scatter(distil_points[:, 0], distil_points[:, 1], color='red', label=f'Student (L{distil_layer_idx})', s=10) # Smaller points
            ax.scatter(gpt2_points[:, 0], gpt2_points[:, 1], color='blue', label=f'Teacher (L{gpt2_layer_idx})', s=10)
            
            # Add token labels (Uncommenting to display labels)
            for i, (x, y) in enumerate(distil_points):
                ax.text(x, y, tokens[i], color='red', fontsize=8)
            for i, (x, y) in enumerate(gpt2_points):
                ax.text(x, y, tokens[i], color='blue', fontsize=8)
                
            ax.set_title(f"Layer {distil_layer_idx} (Student) vs Layer {gpt2_layer_idx} (Teacher)")
            ax.legend()
            ax.set_xticks([]) # Hide axes ticks for clarity
            ax.set_yticks([])
            plot_idx += 1 # Move to the next subplot

        except Exception as e:
            print(f"Error processing layer pair ({distil_layer_idx}, {gpt2_layer_idx}): {str(e)}")
            if plot_idx < len(axes):
                ax = axes[plot_idx]
                ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center')
                ax.set_title(f"Error in Pair ({distil_layer_idx}, {gpt2_layer_idx})")
                plot_idx += 1

    # Hide any unused subplots
    for i in range(plot_idx, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    try:
        plt.savefig(save_path, bbox_inches='tight', dpi=150) # Save the plot
        print(f"t-SNE plot saved to {save_path}")
    except Exception as e:
        print(f"Error saving t-SNE plot to {save_path}: {str(e)}")
    plt.close(fig) # Close the figure to free memory
# --- End t_SNE function definition ---


def main():
    # 初始化分布式进程组
    dist.init_process_group(backend='nccl', init_method='env://')
    args = get_args()
    
    local_rank = int(os.getenv("LOCAL_RANK", "0"))  # 默认值为 0
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    
    # 使用 args.base_path 构建相对路径
    base_path = args.base_path if args.base_path else "."
    
    # Initiailize the model as well trained distill gpt2
    teacher_model, tokenizer = getgpt2(device)
    
    # student_model, tokenizer = get_distillgpt2(device) # <-- 取消注释这行，从 distilgpt2 初始化
    student_model, tokenizer = get_distillgpt2_with_reset(device) #<-- 使用这行替代，如果需要重置层
    
    # student_model = get_student_model(teacher_model, device, args) # <-- 或者如果你想用这个函数初始化
    
    # 下面这几行注释掉，因为是第一次运行，没有预训练模型可加载
    # trained_model_path = os.path.join(base_path, "output", "distillated_student_KL", "student_model_well-trained05000distill_step")
    # student_model, tokenizer = get_trained(trained_model_path, device)
    
    #student_model, tokenizer = get_distillgpt2_with_reset(device)
    # Initialize the teacher model as well trained gpt2-dolly
    if dist.get_rank() == 0:
        print("teacher model -- Check !")
    
    
    world_size = dist.get_world_size()
    
    # Restore manual optimizer and scheduler creation HERE
    student_optimizer = optim.AdamW(student_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    student_scheduler = CosineAnnealingLR(student_optimizer, T_max=1000)
    
    if dist.get_rank() == 0:
        print("student model -- Check!")
    
    
    critics, optimizers, schedulars = get_critics_and_optimizers(student_model.config.n_layer, student_model.config.n_embd, device)
    if dist.get_rank() == 0:
        print("Critics -- Check!")
        
    tokenizer.pad_token = tokenizer.eos_token
    teacher_model.config.pad_token_id = tokenizer.pad_token_id
    student_model.config.pad_token_id = tokenizer.pad_token_id
    
    # Revert to DDP for student model
    student_model = DDP(student_model, device_ids=[local_rank], output_device=local_rank)
    # Keep DDP for teacher and critics
    teacher_model = DDP(teacher_model, device_ids=[local_rank], output_device=local_rank)
    for i, each_critic in enumerate(critics):
        critics[i] = DDP(each_critic, device_ids=[local_rank], output_device=local_rank)

    # Remove DeepSpeed initialization block
    # model_engine, student_optimizer, _, student_lr_scheduler = deepspeed.initialize(
    #     args=args,                  # Pass runtime args
    #     model=student_model,        # Pass the raw student model
    #     model_parameters=student_model.parameters() # Explicitly pass parameters
    # )
    # critic_optimizers = optimizers 
    # critic_schedulers = schedulars


    # --- Add Model Precision Check (After DDP Wrapping) ---
    if dist.get_rank() == 0:
        print("\n--- Model Precision Check (After DDP Wrapping) ---")
        print(f"Command line args.dtype: {args.dtype}") # Print the intended dtype
        try:
            # Check dtype using the DDP module
            param_dtype = student_model.module.transformer.wte.weight.dtype
            print(f"Student model parameter dtype (e.g., embeddings): {param_dtype}")
            # Keep checking teacher via DDP module
            teacher_param_dtype = teacher_model.module.transformer.wte.weight.dtype
            print(f"Teacher model parameter dtype (e.g., embeddings): {teacher_param_dtype}")
        except AttributeError as e:
            print(f"Could not access model parameter dtype for checking: {e}")
        except Exception as e:
             print(f"An error occurred during model precision check: {e}")

        # Check critic dtype after DDP wrapping
        try:
            if critics: # Ensure critics list is not empty
                 # Check critic via DDP module
                 critic_param_dtype = critics[0].module.layer_1[0].weight.dtype # Example path, might need adjustment
                 print(f"Critic model parameter dtype (e.g., layer 1 weight): {critic_param_dtype}")
            else:
                 print("No critics to check.")
        except Exception as e:
            print(f"Could not check critic parameter dtype: {e}")
        print("-----------------------------------------------\n")
    # --- End Model Precision Check ---

    
    if dist.get_rank() == 0:
        print("Dataset -- Check!")
    rng_sample = random.Random(args.seed)
    #data = LMTrainDataset(args, tokenizer, args.data_dir, "train", args.train_num, args.train_ratio, rng_sample)
    #test_data_path = "/home/zyq/experiment_Wasserstein_Distillation/mains/processed_data/dolly/full/gpt2/valid.jsonl" 
    #test_dataset = Dataset.from_json(test_data_path)
    #print("Loading test dataset")
    #test_dataset = test_dataset.map(preprocess_function)

    """dataloader = DataLoader(data, batch_size=32,  # 可根据显存大小调整
                            shuffle=True,  # 训练时随机打乱数据
                            collate_fn = data.collate,
                            num_workers=4,  # 提高数据加载速度
                            pin_memory=True,  # GPU 训练时启用，提升数据传输效率
                            )"""
                            
    # dataset_path = "/home/zyq/experiment_Wasserstein_Distillation/mains/opentext_dataset"  # 替换为本地保存的路径
    dataset_path = os.path.join(base_path, "data", "openwebtext") # 使用相对路径
    dataset = load_from_disk(dataset_path)
    # Original line using the full dataset (commented out as backup)
    # full_train_dataset = dataset["train"] 
    # Original line causing error (commented out)
    # test_dataset_full = dataset.select(range(len(dataset) - 10000, len(dataset))) # Keep test set separate for now

    # --- New code: Limit training dataset size using args.train_num ---
    train_split = dataset["train"] # Get the 'train' split dataset object
    if args.train_num > 0 and args.train_num < len(train_split):
        print(f"Using a subset of the training data: {args.train_num} samples.")
        # Create a subset containing only the first train_num samples
        limited_train_dataset = train_split.select(range(args.train_num))
    else:
        print("Using the full training dataset.")
        limited_train_dataset = train_split # Use the full dataset if train_num is invalid or not set

    # Create the test set from the end of the 'train' split
    print(f"Creating test set from the last 10000 samples of the 'train' split.")
    test_dataset_source = dataset["train"] # Explicitly get the train split again for clarity or use train_split
    test_dataset_indices = range(len(test_dataset_source) - 10000, len(test_dataset_source))
    test_dataset_selected = test_dataset_source.select(test_dataset_indices)
    # --- New code end ---

    # Original line creating OpenWebTextDataset instance from full dataset (commented out as backup)
    # data = OpenWebTextDataset(dataset=full_train_dataset,tokenizer=tokenizer,max_length=512) 
    
    # Create data instance using the potentially limited dataset
    data = OpenWebTextDataset(dataset=limited_train_dataset,tokenizer=tokenizer,max_length=512)
    # Use the selected test_dataset for the test instance
    test_dataset = OpenWebTextDataset(dataset=test_dataset_selected,tokenizer=tokenizer,max_length=512) 
    sampler = DistributedSampler(data, shuffle=True) # Sampler should be based on the limited 'data'
    dataloader = DataLoader(data, batch_size=args.batch_size,  # 从 args 读取 batch size
                            shuffle=False,  # sampler 存在时必须为 False
                            sampler=sampler,
                            num_workers=args.num_workers,  # 从 args 读取 num workers
                            drop_last=True,
                            pin_memory=True,  # GPU 训练时启用，提升数据传输效率
                            )
    
    
       
    if dist.get_rank() == 0:
        print("Start training...")
        print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
        print(f"Available GPUs: {torch.cuda.device_count()}")
    
    if dist.get_rank() == 0:
        print("Evaluating Student score")
    ppl = calculate_perplexity(student_model, tokenizer, test_dataset, device)
    if dist.get_rank() == 0:
        print(f"Student model Perplexity on validation set: {ppl}")
    CFG = cfg("distill_step") # distill_step kl_MSE_step
    
    train_step(
        CFG,
        schedulars, # Pass original schedulers list
        student_scheduler, # Pass manually created scheduler
        test_dataset,
        data_loader=dataloader,
        student_model=student_model, # Pass the DDP wrapped student model
        teacher_model=teacher_model,
        tokenizer = tokenizer,
        critics=critics,
        optimizers=optimizers, # Pass original optimizers list (incl. student)
        student_optimizer = student_optimizer, # Pass manually created student optimizer
        num_epochs=args.epochs, 
        device=device,
        args=args
    )
    
    print("Training completed.")
    
    # --- Add t-SNE plotting after training ---
    if args.local_rank == 0 and args.generate_tsne_plot: # Only run on the main process AND if flag is set
        print("Generating final t-SNE plot...")
        try:
            # Define the output directory for plots relative to base_path
            base_path = args.base_path if args.base_path else "."
            plot_dir = os.path.join(base_path, "output", "fig")
            os.makedirs(plot_dir, exist_ok=True) # Create the directory if it doesn't exist
            save_file_path = os.path.join(plot_dir, "final_tsne_plot.png")

            # Ensure models are on the correct device and unwrapped if necessary for plotting
            # The models passed to train_step are already DDP wrapped. 
            # t_SNE function handles unwrapping with .module
            
            # Call the t-SNE function
            t_SNE(student=student_model, 
                  teacher=teacher_model, 
                  tokenizer=tokenizer, 
                  device=device, 
                  save_path=save_file_path)
        except Exception as e:
            print(f"Error generating or saving t-SNE plot: {e}")
    # --- End t-SNE plotting ---
    
    dist.destroy_process_group()
    
    # 探索不同维度的WD蒸馏？？？？？？？？？？？？？？？？？？
if __name__ == "__main__":
    main()
