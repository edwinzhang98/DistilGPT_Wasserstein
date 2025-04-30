import os
from datasets import load_dataset
import logging
import time

# --- 配置 --- Configuration ---
# 如果你需要使用 Hugging Face 镜像，请设置环境变量
# If you need to use the Hugging Face mirror, please set environment variables
HF_MIRROR_ENDPOINT = 'https://hf-mirror.com'
os.environ['HF_ENDPOINT'] = HF_MIRROR_ENDPOINT
logging.info(f"设置 Hugging Face Endpoint 为: {HF_MIRROR_ENDPOINT}") # Set Hugging Face Endpoint to

# 获取当前脚本所在的目录，作为项目根目录
# Get the directory where the current script is located as the project root directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data') # 定义统一的数据存储目录 # Define a unified data storage directory

# 定义需要下载的数据集及其目标保存路径 (使用相对路径)
# Define the datasets to be downloaded and their target save paths (using relative paths)
DATASETS_TO_DOWNLOAD = {
    "openwebtext": {
        # 注意: Skylion007/openwebtext 可能很大且下载较慢，也有其他版本如 'openwebtext'
        # Note: Skylion007/openwebtext can be large and slow to download, other versions like 'openwebtext' exist
        "identifier": "Skylion007/openwebtext",
        "config_name": None, # OpenWebText 通常没有特定的配置名 # OpenWebText usually doesn't have a specific configuration name
        # "save_path": "/home/zyq/experiment_Wasserstein_Distillation/mains/opentext_dataset"
        "save_path": os.path.join(DATA_DIR, "openwebtext")
    },
    "cnn_dailymail": {
        "identifier": "cnn_dailymail",
        "config_name": "3.0.0", # CNN/DM 通常需要指定版本号 # CNN/DM usually requires specifying the version number
        # "save_path": "/home/zyq/experiment_Wasserstein_Distillation/experiment_DistillBERT/cnn_data/CNN"
        "save_path": os.path.join(DATA_DIR, "cnn_dailymail")
    },
    # 可选: 如果你需要 Dolly 数据集（原始格式），取消下面的注释
    # Optional: If you need the Dolly dataset (original format), uncomment the following lines
    # "dolly": {
    #     "identifier": "databricks/dolly-v2-12k",
    #     "config_name": None,
    #     # 注意：这个路径保存的是原始数据，不是 .bin/.idx 格式
    #     # Note: This path saves the raw data, not in .bin/.idx format
    #     # "save_path": "/home/zyq/experiment_Wasserstein_Distillation/mains/raw_data/dolly"
    #     "save_path": os.path.join(DATA_DIR, "dolly_raw")
    # }
}

# 设置日志记录 # Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 辅助函数 --- Helper Functions ---
def download_and_save(dataset_key, details):
    """下载指定的数据集并保存到本地磁盘""" # Download the specified dataset and save it to local disk
    identifier = details["identifier"]
    save_path = details["save_path"]
    config_name = details.get("config_name") # 如果有配置名，则获取 # Get the config name if it exists

    logging.info(f"--- 开始处理数据集: {dataset_key} ({identifier}) ---") # Starting processing dataset

    # 确保目标目录存在，如果不存在则创建
    # Ensure the target directory exists, create it if it doesn't
    try:
        # os.makedirs 会创建所有必需的父目录 # os.makedirs will create all necessary parent directories
        os.makedirs(save_path, exist_ok=True)
        logging.info(f"已确认目录存在或已创建: {save_path}") # Confirmed directory exists or created
    except OSError as e:
        logging.error(f"创建目录 {save_path} 时出错: {e}") # Error creating directory
        return # 如果目录创建失败，则跳过此数据集 # Skip this dataset if directory creation fails

    # 检查数据是否已存在（可选，save_to_disk 会覆盖）
    # Check if data already exists (optional, save_to_disk will overwrite)
    # if os.path.exists(os.path.join(save_path, "dataset_info.json")):
    #     logging.info(f"数据集 '{identifier}' 似乎已存在于 {save_path}。跳过下载。") # Dataset '{identifier}' seems to exist at {save_path}. Skipping download.
    #     return

    try:
        start_time = time.time()
        logging.info(f"开始下载 '{identifier}'" + (f" (配置: '{config_name}')" if config_name else "") + "...") # Starting download

        # 加载数据集 (会自动处理下载)
        # Load dataset (download is handled automatically)
        # 如果有 config_name 才传递它
        # Pass config_name only if it exists
        if config_name:
            # trust_remote_code=True 在加载某些需要执行代码的数据集脚本时可能是必要的
            # trust_remote_code=True might be necessary when loading dataset scripts that require executing code
            dataset = load_dataset(identifier, config_name, trust_remote_code=True)
        else:
            dataset = load_dataset(identifier, trust_remote_code=True)

        download_time = time.time() - start_time
        logging.info(f"数据集 '{identifier}' 下载完成，耗时: {download_time:.2f} 秒。") # Dataset '{identifier}' download complete, took: {download_time:.2f} seconds.

        # 保存到磁盘 # Save to disk
        start_time = time.time()
        logging.info(f"开始保存数据集到本地路径: {save_path}") # Starting to save dataset to local path
        dataset.save_to_disk(save_path)
        save_time = time.time() - start_time
        logging.info(f"成功保存 '{identifier}' 到 {save_path}，耗时: {save_time:.2f} 秒。") # Successfully saved '{identifier}' to {save_path}, took: {save_time:.2f} seconds.

    except Exception as e:
        logging.error(f"下载或保存数据集 '{identifier}' 失败。错误: {e}") # Failed to download or save dataset '{identifier}'. Error: {e}
        logging.error(f"请检查数据集标识符、网络连接、磁盘空间以及对路径 {save_path} 的写入权限。") # Please check dataset identifier, network connection, disk space, and write permissions for path {save_path}.
        if "dolly" in identifier:
            logging.warning("注意: 此处下载的 Dolly 数据集是原始格式。"
                            "你的项目可能需要使用额外的脚本将其预处理成 '.bin'/''.idx' 格式。") # Note: The Dolly dataset downloaded here is in raw format. Your project might need additional scripts to preprocess it into '.bin'/'.idx' format.

# --- 主执行逻辑 --- Main Execution Logic ---
if __name__ == "__main__":
    logging.info("开始执行数据集获取脚本。") # Starting dataset acquisition script.

    for key, details in DATASETS_TO_DOWNLOAD.items():
        download_and_save(key, details)

    logging.info("数据集获取脚本执行完毕。") # Dataset acquisition script finished.
    logging.info("请检查指定路径下数据集是否已正确下载。") # Please check if the datasets are downloaded correctly in the specified paths.
    logging.warning("请注意，如果下载了 Dolly 数据集，它需要额外的预处理才能生成 lm_datasets.py 所需的 '.bin'/''.idx' 文件。") # Please note that if the Dolly dataset was downloaded, it requires additional preprocessing to generate the '.bin'/'.idx' files needed by lm_datasets.py.