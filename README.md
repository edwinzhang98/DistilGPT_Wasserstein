# DistillGPT2 based on Wasserstein Distance

This project aims to compare the effects of using Mean Squared Error (MSE) and Wasserstein Distance (WD) as different loss functions for GPT-2 knowledge distillation. The experimental results indicate that using Wasserstein Distance as the loss function allows the model to better learn the data distribution features at each layer.

## Repository Structure

This repository contains the following main parts:

- **`Code/`**: Contains all project-related source code, configuration files, and scripts.
  - **Main Scripts and Configuration Files:**
    - `data_download.py`: Script used to download the required datasets. This is the first step to run the project.
    - `wd_stu_base.sh`: The main execution script, used to start the training process. Contains configurable parameters (e.g., learning rate, batch size).
    - `arguments.py`: Defines and parses command-line arguments required for script execution.
    - `cnn_dataset.py`, `indexed_dataset.py`, `lm_datasets.py`: Used for loading and preprocessing datasets of different types or formats.
    - `critic.py`: Implements the Critic network (or related discriminator) required for Wasserstein Distance calculation.
    - `distributed_indexed.py`: Handles data loading and indexing in a distributed training environment.
    - `ds_config_bf16.json`: DeepSpeed configuration file for setting up distributed training strategies and BF16 mixed-precision training.
    - `evaluate_.py`: Script used for evaluating model performance after training.
    - `train_related.py`: Contains core training logic, including the training loop, optimizer setup, and loss calculation.
    - `utils.py`: Contains reusable utility functions for the project, such as logging, model saving/loading, etc.
    - `try_sh.py`, `try_sh_MSE.py`: Potentially backup run scripts for specific experiments (e.g., using only MSE loss) or debugging purposes.
    - `.gitignore`: Specifies files and directories that should be ignored by Git version control (e.g., `output/`, `data/`, temporary files).
  - **Data and Output:**
    - `data/`: Stores downloaded or processed datasets. The `.gitkeep` file ensures this directory is tracked by Git even if empty.
    - `output/`: Stores experiment results.
      - `output_test.log`: Log file from a complete experimental run (on a 3xL20 GPU environment).
      - Other `.log` files: Logs generated during debugging (on an A100 80G environment).
      - `fig/`: May contain visualization results like T-SNE plots.
    - `archive/`: (Potentially exists) Contains old or archived scripts/notebooks, e.g., `test1.ipynb`.
- **`Individual-Final-Project-Report/`**: Contains the final individual project report (`Changhong-Zhang-Final-Project.pdf`).
- **`NLP_Final_Project_Instructions.txt`**: The project instruction document provided by the course.

## How to Run

After cloning this project to a new environment, please follow these steps to run it:

1.  **Environment Setup**: (If necessary) Create and activate a virtual environment, then install the required dependencies. It is recommended to create a `requirements.txt` file based on the code.
    ```bash
    pip install -r Code/requirements.txt
    ```
2.  **Download Data**: Navigate into the `Code` directory and run the data download script:
    ```bash
    cd Code
    python data_download.py
    cd .. 
    ```
3.  **Execute Training**: Run the main `wd_stu_base.sh` script to start the training:
    ```bash
    bash Code/wd_stu_base.sh
    ```
4.  **Modify Parameters**: To adjust hyperparameters (e.g., learning rate, batch size, model configuration), directly edit the `Code/wd_stu_base.sh` file or the scripts it calls.

## Report

The final individual project report is located in the `Individual-Final-Project-Report/` folder. 
