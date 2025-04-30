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
from train_related import save_student_model, example_gen, top_p_sampling, get_student_model, get_gpt2dolly, get_distillgpt2, getgpt2, calculate_perplexity, preprocess_function, evaluate_model_with_rouge
from cnn_dataset import CNNDMDistillationDataset
from torch.nn.parallel import DistributedDataParallel as DDP


def get_critics_and_optimizers(n, dim, device, lr=1e-4):
    # n := number of layers need distillation
    # dim := embedding dimension 
    # lr := learning rate
    critics = []
    optimizers = []
    schedulers = []
    
    for i in range(n):
        # 定义 Critic
        # Define Critic
        new_critic = critic(dim)
        new_critic = new_critic.to(device)
        critics.append(new_critic)
        
        # 定义对应的 Optimizer
        # Define the corresponding Optimizer
        optimizer = optim.RMSprop(new_critic.parameters(), lr=lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=1000)
        optimizers.append(optimizer)
        schedulers.append(scheduler)
        
    return critics, optimizers, schedulers


import os


def token_level_distill(args, model_data, student_model, teacher_model, critics, critic_optimizer, student_optimizer, student_layer, teacher_layer, generated_ids, attention_mask):
    for _ in range(args.max_input_len - model_data["input_ids"].shape[1]):

        # 前向传播：教师和学生模型
        # Forward pass: Teacher and student models
        with torch.no_grad():
            teacher_output = teacher_model(input_ids=generated_ids, attention_mask=attention_mask, output_hidden_states=True)
        student_output = student_model(input_ids=generated_ids, attention_mask=attention_mask, output_hidden_states=True)

        #使用教师logits
        # Use teacher logits
        teacher_logits = teacher_output.logits
        next_token_logits = teacher_logits[:,-1,:]
        # 学生模型的 logits
        # Student model's logits
        student_logits = student_output.logits
        student_next_token_logits = student_logits[:, -1, :]
                        
        # 计算 KL loss
        # Calculate KL loss
        teacher_probs = F.softmax(next_token_logits, dim=-1)
        student_probs = F.log_softmax(student_next_token_logits, dim=-1)
        kl_loss = F.kl_div(student_probs, teacher_probs, reduction='batchmean')

        # Top-p + Temperature
        next_token = top_p_sampling(next_token_logits, top_p=0.9, temperature=0.8)
        # 将生成的 token 添加到输入序列
        # Add the generated token to the input sequence
        next_token = next_token.unsqueeze(1)
        generated_ids = torch.cat([generated_ids, next_token], dim=1)
        #with torch.no_grad():
            # 更新 attention_mask，确保新增的 token 被关注
            # Update attention_mask to ensure the new token is attended to
        new_attention_mask = torch.ones((attention_mask.size(0), 1), device=attention_mask.device)
        attention_mask = torch.cat([attention_mask, new_attention_mask], dim=1)

        # Critic 蒸馏步骤
        # Critic distillation step
        critic_loss, student_loss = critic_step(
                                cur_stu_ly=student_layer,
                                cur_tea_ly=teacher_layer,
                                student_model=student_model,
                                student_output=student_output,
                                target=teacher_output,
                                critics=critics,
                                critic_optimizer=critic_optimizer,
                                student_optimizer=student_optimizer,  # 学生模型的 optimizer 放在最后 # Put student model's optimizer last
                                arg=args
                                )
        # 将 KL loss 加入到学生模型的 loss 中
        # Add KL loss to the student model's loss
        total_loss = student_loss +  kl_loss
        # 反向传播和优化学生模型
        # Backpropagate and optimize the student model
        student_optimizer.zero_grad()
        total_loss.backward()
        student_optimizer.step()
    return total_loss, student_loss, critic_loss, kl_loss
        
        
def distill_step(args, student_model, student_optimizer, teacher_model, critic_list, optimizer_list, input_ids, attention_mask):
    # 前向传播：教师和学生模型
    # Forward pass: Teacher and student models
    with torch.no_grad():
        teacher_output = teacher_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    student_output = student_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    teacher_states = teacher_output.hidden_states
    teacher_states = [teacher_states[(i+1) * 2 - 1] for i in range(12 // 2)]
    student_states = student_output.hidden_states
    
    # 使用教师 logits，添加 temperature
    # Use teacher logits, add temperature
    temperature = 1.2
    teacher_logits = teacher_output.logits / temperature
    student_logits = student_output.logits / temperature

    # 计算 KL loss
    # Calculate KL loss
    teacher_probs = F.softmax(teacher_logits, dim=-1)
    student_probs = F.log_softmax(student_logits, dim=-1)
    kl_loss = F.kl_div(student_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)

    # Critic 蒸馏步骤
    # Critic distillation step
    critic_loss_container = []
    total_stu_loss = 0
    for student_emb, teacher_emb, critic, optimizer_critic in zip(student_states, teacher_states, critic_list, optimizer_list):
        critic_loss, student_loss = critic_step(student_emb, teacher_emb, critic, optimizer_critic, args)
        total_stu_loss += student_loss
        critic_loss_container.append(critic_loss)
                        
    
    # 将 KL loss 加入到学生模型的 loss 中
    # Add KL loss to the student model's loss
    total_loss = student_loss + kl_loss

    # 反向传播和优化学生模型
    # Backpropagate and optimize the student model
    student_optimizer.zero_grad()
    total_loss.backward()
    student_optimizer.step()

    return total_loss, student_loss, critic_loss_container, kl_loss

def kl_MSE_step(args, student_model, student_optimizer, teacher_model, critic_list, optimizer_list, input_ids, attention_mask):
    # 前向传播：教师和学生模型
    # Forward pass: Teacher and student models
    with torch.no_grad():
        teacher_output = teacher_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    student_output = student_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    teacher_states = teacher_output.hidden_states
    teacher_states = [teacher_states[(i+1) * 2 - 1] for i in range(12 // 2)]
    student_states = student_output.hidden_states
    
    # 使用教师 logits，添加 temperature
    # Use teacher logits, add temperature
    temperature = 1.2
    teacher_logits = teacher_output.logits / temperature
    student_logits = student_output.logits / temperature

    # 计算 KL loss
    # Calculate KL loss
    teacher_probs = F.softmax(teacher_logits, dim=-1)
    student_probs = F.log_softmax(student_logits, dim=-1)
    kl_loss = F.kl_div(student_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)
    
    # MSE LOSS
    mse = 0
    for student_emb, teacher_emb in zip(student_states, teacher_states):
        mse += F.mse_loss(student_emb, teacher_emb)
    
    total_loss = mse + kl_loss
        
    # 反向传播和优化学生模型
    # Backpropagate and optimize the student model
    student_optimizer.zero_grad()
    total_loss.backward()
    student_optimizer.step()

    return None, None, mse, kl_loss



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
    # Start training the Critic
    for _ in range(arg.critic_time):
        teacher_score = critic(teacher_emb)
        student_output_detached = student_emb.detach()
        student_score_critic = critic(student_output_detached)
        # 判别器的损失：最大化教师评分，最小化学生评分
        # Discriminator loss: Maximize teacher score, minimize student score
        critic_loss = -(torch.mean(teacher_score) - torch.mean(student_score_critic))
        gp = gradient_penalty(critic, teacher_emb, student_output_detached)
        critic_loss += arg.lambda_gp * gp
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()
    # 重新计算 student_score，这次不分离计算图
    # Recalculate student_score, this time without detaching the computation graph
    student_score = critic(student_emb)

    # 学生模型的损失：最小化学生评分
    # Student model loss: Minimize student score
    student_loss = -torch.mean(student_score)
    
    return critic_loss.item(), student_loss
    


def train_step(schedulars, student_scheduler, test_dataset, data_loader, student_model, teacher_model, tokenizer, critics, optimizers, student_optimizer, num_epochs, device, args):
    # Critic -> compute Wasserstein Distance
    # 设置训练模式
    # Set training mode
    teacher_model.eval()
    student_model.train()
    for each_critic in critics:
        each_critic.train()    
        
    # 得到需要蒸馏的总层数
    # Get the total number of layers to be distilled
    n_layers = student_model.config.n_layer
    n_layers_teacher = teacher_model.config.n_layer
    
    # 开始训练：逐层进行蒸馏
    # Start training: Distill layer by layer
        
    for epoch in range(num_epochs):              
        
        for step, batch in enumerate(data_loader):
            # 初始输入 (复制一份作为生成用，避免修改原数据)
            # Initial input (copy one for generation to avoid modifying original data)
            # 数据移至 GPU，遍历数据批次
            # Move data to GPU, iterate through data batches
            #model_data, no_model_data, gen_data = batch
            model_data = batch
            input_ids = model_data["input_ids"].to(device)
            attention_mask = model_data["attention_mask"].to(device)
            generated_ids = input_ids.clone()
                
            # 自回归生成/不自回归
            # Autoregressive generation / Non-autoregressive
            time1 = time.time()
            total_loss, student_loss, critic_loss_container, kl_loss = kl_MSE_step(args, student_model, student_optimizer, teacher_model, critics, optimizers, input_ids, attention_mask)
            # 每 20 步打印一次损失
            # Print loss every 50 steps
            if step % 50 == 0:
                time2 = time.time()
                epsilon = time2 - time1
                if student_loss is None:
                    print(f" Epoch [{epoch}/{num_epochs}] | Step [{step}] | MSE loss: {critic_loss_container.item():.4f} |Kl loss: {kl_loss:.4f}| Time: {epsilon:.4f}")
                else:
                    print(f" Epoch [{epoch}/{num_epochs}] | Step [{step}] | Student Loss: {student_loss.item():.3f} | C1L: {critic_loss_container[0]:.3f}, C2L: {critic_loss_container[1]:.3f}, C3L: {critic_loss_container[2]:.3f}, C4L: {critic_loss_container[3]:.3f}, C5L: {critic_loss_container[4]:.3f}, C6L: {critic_loss_container[5]:.3f}, | Kl loss: {kl_loss:.4f}| Time: {epsilon:.4f}")
                         
            # 每 400 步生成一次示例
            if step % 400 == 0:
                example_gen(student_model, teacher_model, tokenizer, device)
            
        if epoch % 1 == 0:
            ppl = calculate_perplexity(student_model, tokenizer, test_dataset)
            print(f"Perplexity on Dolly validation set: {ppl}")
            
        student_scheduler.step()
        for each_scheduler in schedulars:
            each_scheduler.step()


                    
                    
                    
                    
    student_save_path = os.path.join(base_path, "output", "distillated_student_KL")
    os.makedirs(student_save_path, exist_ok=True) #确保目录存在
    save_student_model(student_model, student_save_path, cur_ly='well-trained')



def main():
    # 初始化分布式进程组
    # Initialize distributed process group
    dist.init_process_group(backend='nccl', init_method='env://')
    args = get_args()
    
    local_rank = int(os.getenv("LOCAL_RANK", "0"))  # 默认值为 0
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    
    # Initiailize the model as well trained distill gpt2
    student_model, tokenizer = get_distillgpt2(device)
    student_optimizer = optim.AdamW(student_model.parameters(), lr=1e-5)
    student_scheduler = CosineAnnealingLR(student_optimizer, T_max=1000)
    print("student model set up already")
    
    # Initialize the teacher model as well trained gpt2-dolly
    teacher_model, tokenizer = get_gpt2dolly(device)
    print("teacher model set up already")
    critics, optimizers, schedulars = get_critics_and_optimizers(student_model.config.n_layer, student_model.config.n_embd, device)
    print("Critics are set up already")
        
    tokenizer.pad_token = tokenizer.eos_token
    teacher_model.config.pad_token_id = tokenizer.pad_token_id
    student_model.config.pad_token_id = tokenizer.pad_token_id
    
    print("Setting up Dataset")
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
                            
    dataset_path = os.path.join(base_path, "data", "cnn_dailymail") # 使用相对路径
    dataset = load_from_disk(dataset_path)
    data = CNNDMDistillationDataset(dataset=dataset["train"],tokenizer=tokenizer,max_length=512)
    test_dataset = CNNDMDistillationDataset(dataset=dataset["test"],tokenizer=tokenizer,max_length=512)
    dataloader = DataLoader(data, batch_size=32,  # 可根据显存大小调整
                            shuffle=True,  # 训练时随机打乱数据
                            num_workers=4,  # 提高数据加载速度
                            pin_memory=True,  # GPU 训练时启用，提升数据传输效率
                            )
    
   
    print("Dataset is set up already")
    
    print("Starting training...")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    
    print("Evaluating Teacher score")
    ppl = calculate_perplexity(teacher_model, tokenizer, test_dataset)
    print(f"Teacher model Perplexity on Dolly validation set: {ppl}")
    print("Evaluating Student score")
    ppl = calculate_perplexity(student_model, tokenizer, test_dataset)
    print(f"Perplexity on Dolly validation set: {ppl}")
    
    train_step(
        schedulars,
        student_scheduler,
        test_dataset,
        data_loader=dataloader,
        tokenizer = tokenizer,
        student_model=student_model,
        teacher_model=teacher_model,
        critics=critics,
        optimizers=optimizers,
        student_optimizer = student_optimizer,
        num_epochs=20,
        device=device,
        args=args
    )
    
    print("Training completed.")
    
    
    dist.destroy_process_group()
    
    # 探索不同维度的WD蒸馏？？？？？？？？？？？？？？？？？？
if __name__ == "__main__":
    main()

