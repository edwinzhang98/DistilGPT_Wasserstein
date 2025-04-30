from arguments import get_args
from lm_datasets import LMTrainDataset
from transformers import GPT2Config, GPT2LMHeadModel
from utils import get_tokenizer, get_model, get_teacher_model
from critic import critic
import evaluate
import random
import torch
import os
from torch.utils.data import Subset
import torch.distributed as dist
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import time
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer, AutoModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import torch
from datasets import Dataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from torch.utils.data.distributed import DistributedSampler

def example_gen(student, teacher, tokenizer, device):
    # 生成文本
    prompt = "How to make a cake?"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # 教师模型生成文本
    teacher_output = teacher.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=50,
        pad_token_id=tokenizer.pad_token_id,
        num_return_sequences=1,  # 返回的生成样本数量
        temperature=1.0,  # 控制生成的随机性
        top_k=50,  # 只考虑 top-k 高概率词汇
        top_p=0.95,  # nucleus sampling 的概率阈值
        do_sample=True  # 允许采样
    )
    
    student_output = student.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=50,
        pad_token_id=tokenizer.pad_token_id,
        num_return_sequences=1,  # 返回的生成样本数量
        temperature=1.0,  # 控制生成的随机性
        top_k=50,  # 只考虑 top-k 高概率词汇
        top_p=0.95,  # nucleus sampling 的概率阈值
        do_sample=True  # 允许采样
    )

    # 解码输出
    student_text = tokenizer.decode(student_output[0], skip_special_tokens=True)
    print("Generated Text by Student Model:")
    print(student_text)

    # 解码输出
    teacher_text = tokenizer.decode(teacher_output[0], skip_special_tokens=True)
    print("Generated Text by Teacher Model:")
    print(teacher_text)
    

# 加载 GPT2-Dolly 模型
def get_gpt2dolly(device):
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("CyrexPro/gpt2-finetuned-cnn_dailymail")  # 带语言模型头
    model = model.to(device)
    model.eval()  # 设置为评估模式
    return model, tokenizer

def get_trained(path, device):
    model = AutoModelForCausalLM.from_pretrained(path)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = model.to(device)
    return model, tokenizer


def getgpt2(device):
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")  # 带语言模型头
    model = model.to(device)
    model.eval()  # 设置为评估模式
    return model, tokenizer

# 加载 DistilGPT2 模型
def get_distillgpt2(device):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('distilgpt2')  # 使用 GPT2LMHeadModel 替换 GPT2Model abdulmanaam/distillgpt2-sentiment-detection
    model = model.to(device)
    model.train()  # 设置为评估模式
    return model, tokenizer


def reset_layer_parameters(model, layer_idx):
    target_layer = model.transformer.h[layer_idx]
    for module in target_layer.modules():
        if hasattr(module, 'weight') and module.weight is not None:
            print(f"重置权重: {module}")
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if hasattr(module, 'bias') and module.bias is not None:
            print(f"重置偏置: {module}")
            torch.nn.init.zeros_(module.bias)
            
def check_layer_parameters(model, layer_idx):
    target_layer = model.transformer.h[layer_idx]
    for name, param in target_layer.named_parameters():
        print(f"参数 {name}: 均值 {param.mean().item()}, 标准差 {param.std().item()}")
        
def get_distillgpt2_with_reset(device, reset_layer_idx=0):
    """
    加载 DistilGPT2 模型，并选择重置某一层参数。
    Args:
        device: 模型加载到的设备（如 'cuda' 或 'cpu'）。
        reset_layer_idx: 要重置的层索引（默认为 None，不重置任何层）。
    Returns:
        model, tokenizer: 经过初始化的模型和分词器。
    """
    # 加载模型和分词器
    tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
    model = GPT2LMHeadModel.from_pretrained('distilgpt2')
    model = model.to(device)

    # 如果指定了重置层，则重置
    if reset_layer_idx is not None:
        print(f"重置第 {reset_layer_idx} 层的参数...")
        reset_layer_parameters(model, reset_layer_idx)    
    model.train()  # 设置模型为训练模式
    return model, tokenizer


def get_student_model(teacher_model, device, args):
    model_name = args.teacher_ckpt_name
    
    student_model_path = "kk" #"/home/zyq/experiment_Wasserstein_Distillation/mains/distillated_student/student_model_well-trained_KL"
    if os.path.exists(student_model_path):
        print("Loading trained student model from:", student_model_path)
        model = GPT2LMHeadModel.from_pretrained(student_model_path).to(device)
        print("Trained student model loaded successfully.")
        return model
    
    # 加载 GPT-2 XL 的默认配置
    original_config = teacher_model.config

    # 修改层数为 1/3（GPT-2 XL 默认有 48 层）
    modified_config = GPT2Config(
        n_layer=original_config.n_layer // 2,  # 调整层数为 1/3
        n_embd=original_config.n_embd,        # 保持嵌入维度
        n_head=original_config.n_head,        # 保持注意力头数量
        vocab_size=original_config.vocab_size # 保持词汇表大小
    )
    
    # 加载学生模型权重
    print('loading student model')
    model = GPT2LMHeadModel(modified_config)
    print("done")

    
    # 从教师模型加载部分权重
    teacher_state_dict = teacher_model.state_dict()
    student_state_dict = model.state_dict()
    
    # 遍历学生模型的层，将对应的教师模型权重拷贝过来
    for i in range(modified_config.n_layer):
        teacher_idx = i * 2
        student_layer_name = f"transformer.h.{i}"  # 学生模型层名
        teacher_layer_name = f"transformer.h.{teacher_idx}"  # 教师模型层名（假设按顺序直接映射）
        if (
            f"{student_layer_name}.attn.c_attn.weight" in student_state_dict and
            f"{teacher_layer_name}.attn.c_attn.weight" in teacher_state_dict
        ):
            student_state_dict[f"{student_layer_name}.attn.c_attn.weight"] = \
                teacher_state_dict[f"{teacher_layer_name}.attn.c_attn.weight"]
            student_state_dict[f"{student_layer_name}.attn.c_proj.weight"] = \
                teacher_state_dict[f"{teacher_layer_name}.attn.c_proj.weight"]
            student_state_dict[f"{student_layer_name}.mlp.c_fc.weight"] = \
                teacher_state_dict[f"{teacher_layer_name}.mlp.c_fc.weight"]
            student_state_dict[f"{student_layer_name}.mlp.c_proj.weight"] = \
                teacher_state_dict[f"{teacher_layer_name}.mlp.c_proj.weight"]
            print(f"Layer {i+1} successfully loaded from teacher layer {teacher_idx+1}.")
        else:
            print(f"Layer {i+1} could not be mapped to teacher layer {teacher_idx+1}.")
    
    # 初始化模型
    model.load_state_dict(student_state_dict, strict=False)
    model = model.to(device) #Send to DEVICE， 此处没有考虑多设备训练
    print(f"Number of layers: {modified_config.n_layer}")
    # 查看参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")

    return model
    


def top_p_sampling(logits, top_p=0.9, temperature=1.0):
    """
    使用 Top-p + Temperature 采样生成下一个 token。
    :param logits: 模型的原始输出 logits (batch_size, vocab_size)
    :param top_p: Top-p (核采样) 的概率阈值
    :param temperature: 温度系数，用于控制生成的多样性
    :return: 采样得到的 token 索引 (batch_size,)
    """
    # Temperature scaling
    logits = logits / temperature

    # 计算 softmax 概率
    probs = F.softmax(logits, dim=-1)

    # 按概率排序，并计算累积概率
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # 保留累积概率 <= top_p 的 token
    sorted_indices_to_remove = cumulative_probs > top_p
    # 确保至少保留一个 token
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = False

    # 将被过滤的 token 的概率置为 0
    for batch_idx in range(logits.size(0)):
        probs[batch_idx, sorted_indices[batch_idx, sorted_indices_to_remove[batch_idx]]] = 0

    # 重新归一化概率并进行采样
    probs = probs / probs.sum(dim=-1, keepdim=True)
    next_token = torch.multinomial(probs, num_samples=1).squeeze(1)

    return next_token


def save_student_model(student_model, base_save_path, cur_ly, model_name=None):
    """
    保存学生模型的权重和配置，允许存储不同名字的模型。
    Args:
        student_model: 训练完成的学生模型。
        base_save_path: 模型保存的根目录。
        model_name: 可选，模型的名称。如果不提供，默认使用时间戳。
    """
    # 生成模型名称（默认使用时间戳）
    if model_name is None:
        model_name = f"student_model_{cur_ly}"
    
    # 完整保存路径
    save_path = os.path.join(base_save_path, model_name)
    
    # 确保路径存在
    os.makedirs(save_path, exist_ok=True)
    
    # 保存模型
    student_model.module.save_pretrained(save_path)
    print(f"Student model saved to {save_path}")
    
    



# 数据预处理函数：将 prompt 和 output 拼接处理为模型的输入和目标
def preprocess_function(sample):
    prompt = sample["prompt"]
    output = sample["output"]

    # 模型的输入为 prompt，目标为 output
    full_input = prompt  # Prompt 中已经包含了指令和任务描述
    target_output = output

    return {"prompt": full_input, "output": target_output}

# 计算Perplexity的函数
def calculate_perplexity(model, tokenizer, test_dataset, device):
    model.eval()
    total_loss = 0
    total_tokens = 0

    sampler = DistributedSampler(test_dataset, shuffle=False)
    dataloader = DataLoader(test_dataset, batch_size=32, sampler=sampler)

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Get loss
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss

            # Weighted by the number of tokens
            total_loss += loss.item() * input_ids.size(1)
            total_tokens += input_ids.size(1)

    # Sync across all GPUs
    total_loss_tensor = torch.tensor(total_loss, device=device, dtype=torch.float64)
    total_tokens_tensor = torch.tensor(total_tokens, device=device, dtype=torch.float64)

    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(total_loss_tensor, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(total_tokens_tensor, op=torch.distributed.ReduceOp.SUM)

    total_loss = total_loss_tensor.item()
    total_tokens = total_tokens_tensor.item()

    # Compute perplexity on rank 0
    perplexity = None
    if torch.distributed.get_rank() == 0:
        if total_tokens > 0:
            perplexity = torch.exp(torch.tensor(total_loss / total_tokens, device=device))
        else:
            perplexity = torch.tensor(float('inf'), device=device)

    # Broadcast perplexity to all ranks
    perplexity_tensor = torch.zeros(1, device=device)
    if perplexity is not None:
        perplexity_tensor[0] = perplexity
    torch.distributed.broadcast(perplexity_tensor, src=0)
    
    model.train()

    return perplexity_tensor.item()





def calculate_rouge(predictions, references):
    """
    计算 ROUGE 分数。
    Args:
        predictions (list of str): 模型生成的预测文本列表。
        references (list of str): 参考文本（ground truth）列表。

    Returns:
        dict: 包含 ROUGE-1, ROUGE-2, 和 ROUGE-L 分数（F1、Precision、Recall）。
    """
    # 加载 ROUGE 评估器
    rouge_metric = evaluate.load("rouge")
    results = rouge_metric.compute(predictions=predictions, references=references)

    # 检查返回值的格式并提取分数
    rouge_scores = {}
    for rouge_type in ["rouge1", "rouge2", "rougeL"]:
        if isinstance(results[rouge_type], dict) and "mid" in results[rouge_type]:
            # 如果是嵌套对象
            rouge_scores[rouge_type] = {
                "precision": results[rouge_type]["mid"].precision,
                "recall": results[rouge_type]["mid"].recall,
                "f1": results[rouge_type]["mid"].fmeasure,
            }
        else:
            # 如果是简单浮点数
            rouge_scores[rouge_type] = results[rouge_type]

    return rouge_scores



def evaluate_model_with_rouge(model, tokenizer, dataset, max_length=512):
    """
    用 ROUGE 评估模型在生成任务上的性能。
    Args:
        model: 用于生成的语言模型。
        tokenizer: 模型的分词器。
        dataloader: 包含 prompt 和 reference 的 DataLoader。
        max_length: 生成的最大长度。

    Returns:
        dict: 包含 ROUGE 分数。
    """
    model.eval().cuda()
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    predictions = []
    references = []
    tokenizer.padding_side = "left"  # GPT-2 默认的 padding 在右侧，这里改为左侧以对齐

    with torch.no_grad():
        for batch in dataloader:
            prompts = batch["prompt"]
            reference_texts = batch["output"]

            # 模型生成预测文本
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to("cuda")
            outputs = model.generate(**inputs, max_length=max_length, num_beams=4)

            # 解码生成文本和参考文本
            decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            decoded_refs = reference_texts

            predictions.extend(decoded_preds)
            references.extend(decoded_refs)

    # 计算 ROUGE 分数
    rouge_scores = calculate_rouge(predictions, references)
    tokenizer.padding_side = "right"
    return rouge_scores



def t_SNE1(student, teacher, tokenizer, device):
    # 2. 准备示例文本
    text = "This is the experiment of Wasserstein Distillation. Hope it works well."
    inputs = tokenizer(text, return_tensors="pt").to(device)

    # 获取字符串形式的每个 token（以便可视化时给散点打上文字）
    gpt2_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    distil_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    # 3. 前向传播，获取所有层的输出
    with torch.no_grad():
        outputs_gpt2 = teacher(**inputs)
        outputs_distilgpt2 = student(**inputs)

    # 这两个列表包含了 embedding 层 + N 层 Transformer 的输出，每层 shape: [batch_size, seq_len, hidden_size]
    hidden_states_gpt2 = outputs_gpt2.hidden_states       # GPT2 共有 13 个 (0~12)
    hidden_states_distilgpt2 = outputs_distilgpt2.hidden_states  # DistilGPT2 共有 7 个 (0~6)

    # 4. 定义要对齐的层关系: 
    #    DistilGPT2 第 i 层 <-> GPT2 第 2*i 层
    #    即 (1, 2), (2, 4), (3, 6), (4, 8), (5, 10), (6, 12)
    layer_pairs = [(1, 2), (2, 4), (3, 6), (4, 8), (5, 10), (6, 12)]

    # 创建多个子图，每个子图画一对层的 t-SNE
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()  # 让它成为一维数组，方便索引

    for idx, (distil_layer_idx, gpt2_layer_idx) in enumerate(layer_pairs):
        # 取出 DistilGPT2 和 GPT2 该层的所有 token 向量（去掉 batch 维度）
        distil_layer_output = hidden_states_distilgpt2[distil_layer_idx][0]  # shape: [seq_len, hidden_size]
        gpt2_layer_output = hidden_states_gpt2[gpt2_layer_idx][0]           # shape: [seq_len, hidden_size]

        # 将二者在第0维拼接，使得总形状变成 (seq_len_distil + seq_len_gpt2, hidden_size)
        # 由于我们输入同一段文本，seq_len_distil 和 seq_len_gpt2 一般是相同或非常接近的
        combined_embeddings = torch.cat([distil_layer_output, gpt2_layer_output], dim=0).cpu().numpy()

        # 5. 对合并后的向量执行 t-SNE
        # 注意：这里的 perplexity 一定要小于样本数（token 数 * 2）
        tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto', perplexity=10)
        tsne_results = tsne.fit_transform(combined_embeddings)

        # 前一半是 DistilGPT2 的 token 的映射，后一半是 GPT2 的 token
        distil_points = tsne_results[:distil_layer_output.shape[0]]
        gpt2_points = tsne_results[distil_layer_output.shape[0]:]

        ax = axes[idx]
        # 6. 在子图上绘制散点
        ax.scatter(distil_points[:, 0], distil_points[:, 1], color='red', label='DistilGPT2')
        ax.scatter(gpt2_points[:, 0],  gpt2_points[:, 1],  color='blue', label='GPT2')

        # 可以给每个点加上对应的 token 文本标签
        for i, (x, y) in enumerate(distil_points):
            ax.text(x, y, distil_tokens[i], color='red', fontsize=8)
        for i, (x, y) in enumerate(gpt2_points):
            # 注意 GPT2 的 tokenizer 与 DistilGPT2 的 tokenizer 可能略有差异
            ax.text(x, y, gpt2_tokens[i], color='blue', fontsize=8)

        ax.set_title(f"DistilGPT2 Layer {distil_layer_idx} vs GPT2 Layer {gpt2_layer_idx}")
        ax.legend()

    plt.tight_layout()
    plt.show()

def t_SNE(student, teacher, tokenizer, device, step, CFG):
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
        outputs_gpt2 = teacher(
            **inputs,
            output_hidden_states=True,
            return_dict=True
        )
        outputs_distilgpt2 = student(
            **inputs,
            output_hidden_states=True,
            return_dict=True
        )
    
    # Verify hidden states are available
    if outputs_gpt2.hidden_states is None or outputs_distilgpt2.hidden_states is None:
        raise ValueError("Hidden states not returned by the models. Check model configurations.")
    
    # Get tokens for visualization
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    
    # Define layer pairs to align
    layer_pairs = [(1, 2), (2, 4), (3, 6), (4, 8), (5, 10), (6, 12)]
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    # Convert hidden states to list if they're tuples
    hidden_states_gpt2 = list(outputs_gpt2.hidden_states)
    hidden_states_distilgpt2 = list(outputs_distilgpt2.hidden_states)
    
    for idx, (distil_layer_idx, gpt2_layer_idx) in enumerate(layer_pairs):
        try:
            # Get layer outputs and move to CPU for t-SNE
            distil_layer_output = hidden_states_distilgpt2[distil_layer_idx][0].cpu()
            gpt2_layer_output = hidden_states_gpt2[gpt2_layer_idx][0].cpu()
            
            # Concatenate embeddings
            combined_embeddings = torch.cat([distil_layer_output, gpt2_layer_output], dim=0).numpy()
            
            # Calculate safe perplexity value
            n_samples = combined_embeddings.shape[0]
            perplexity = min(10, max(2, n_samples - 1))  # Ensure perplexity is at least 2
            
            # Perform t-SNE
            tsne = TSNE(
                n_components=2,
                random_state=42,
                init='pca',
                learning_rate='auto',
                perplexity=perplexity
            )
            tsne_results = tsne.fit_transform(combined_embeddings)
            
            # Split results for both models
            n_tokens = distil_layer_output.shape[0]
            distil_points = tsne_results[:n_tokens]
            gpt2_points = tsne_results[n_tokens:]
            
            # Plot points
            ax = axes[idx]
            ax.scatter(distil_points[:, 0], distil_points[:, 1], color='red', label='DistilGPT2')
            ax.scatter(gpt2_points[:, 0], gpt2_points[:, 1], color='blue', label='GPT2')
            
            # Add token labels
            for i, (x, y) in enumerate(distil_points):
                ax.text(x, y, tokens[i], color='red', fontsize=8)
            for i, (x, y) in enumerate(gpt2_points):
                ax.text(x, y, tokens[i], color='blue', fontsize=8)
                
            ax.set_title(f"DistilGPT2 Layer {distil_layer_idx} vs GPT2 Layer {gpt2_layer_idx}")
            ax.legend()
            
        except Exception as e:
            print(f"Error processing layer pair ({distil_layer_idx}, {gpt2_layer_idx}): {str(e)}")
            ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center')
    
    plt.tight_layout()
    # plt.show() # 注释掉 show，避免阻塞
    
    # 确保输出目录存在
    output_dir = os.path.join(CFG.base_path if CFG.base_path else ".", "output", "output_pics_MSE")
    os.makedirs(output_dir, exist_ok=True)

    # 保存图像到文件而不是显示
    # 之前错误的绝对路径保存
    # if CFG.distill_step == "kl_MSE_step":
    #     plt.savefig("/home/zyq/experiment_Wasserstein_Distillation/mains/output_pics_MSE" + "/t-SNE" + str(step) + ".png")
    # else:
    #     plt.savefig("/home/zyq/experiment_Wasserstein_Distillation/mains/output_pics_MSE" + "/t-SNE" + str(step) + ".png")
    
    # 使用新的相对路径保存（这里不再区分 distill_step，统一保存）
    fig.savefig(os.path.join(output_dir, f"t-SNE_layers_step_{step}.png")) # 保存包含所有子图的大图
    plt.close(fig)  # 关闭图像以释放内存
    
    print(f"t-SNE visualization saved to {os.path.join(output_dir, f't-SNE_layers_step_{step}.png')}")
    
    # 返回模型到训练模式 (这个应该在调用 t_SNE 的函数外部根据需要设置)
    # student.train()
    # teacher.train()