import os
import sys
from huggingface_hub import snapshot_download
from datasets import load_dataset, Dataset

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from model.quantizer import QwenGPTQQuantizer
from src.grpo_trainer import train_r3_quant_grpo
from src.sft_trainer import train_sft_baseline

BASE_MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
RL_METHOD = "GRPO"
QUANT_BITS = 3


def setup_environment():
    print("--- 1. Khởi tạo cấu trúc thư mục ---")
    directories = ["data/science_qa", "weights", "r3_quant_checkpoints", "sft_baseline_checkpoints"]
    for folder in directories:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Đã tạo thư mục: {folder}")

def download_data():
    print("\n--- 2. Đang tải/đọc Dataset ScienceQA ---")
    target_path = "./data/science_qa/validation-00000-of-00001-6c7328ff6c84284c.parquet"
    if not os.path.exists(target_path):
        print("Đang tải dataset từ Hugging Face...")
        dataset = load_dataset("derek-thomas/ScienceQA", split="validation")
        dataset.to_parquet(target_path)
        print(f"Đã lưu dataset tại: {target_path}")
    else:
        print(f"Dataset đã tồn tại tại {target_path}, đang load...")
        dataset = load_dataset("parquet", data_files=target_path, split="train")
    return dataset

def download_model(model_id):
    print(f"\n--- 3. Đang tải Model {model_id} ---")
    model_name = model_id.split("/")[-1]
    local_dir = f"./weights/{model_name}"
    
    if not os.path.exists(os.path.join(local_dir, "config.json")):
        print(f"Đang tải model {model_id} (quá trình này có thể lâu)...")
        snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            revision="main"
        )
        print(f"Model đã được tải về: {local_dir}")
    else:
        print(f"Model đã tồn tại ở: {local_dir}")
    return local_dir

def run_quantization(base_model_dir, dataset_path, bits):
    model_name = os.path.basename(base_model_dir)
    save_dir = f"./weights/{model_name}-GPTQ-Int{bits}"
    
    print(f"\n--- 4. Bắt đầu lượng tử hoá model (GPTQ Int{bits}) ---")
    if not os.path.exists(os.path.join(save_dir, "config.json")):
        print(f"Đang lượng tử hóa và lưu tại: {save_dir}")
        quantizer = QwenGPTQQuantizer(base_model_dir, save_dir, dataset_path)
        quantizer.quantize_and_save(bits=bits)
        print("[SUCCESS] Quá trình lượng tử hóa hoàn tất thành công!")
    else:
        print(f"Model lượng tử hóa đã tồn tại ở: {save_dir}")
        
    return save_dir

def run_rl_training(quant_model_dir, dataset, method):
    print(f"\n--- 5. Bắt đầu quá trình thiết lập LoRA và huấn luyện RL ({method}) ---")
    if method == "GRPO":
        output_dir = "./r3_quant_checkpoints"
        train_r3_quant_grpo(quant_model_dir, dataset, output_dir)
    elif method == "SFT":
        output_dir = "./sft_baseline_checkpoints"
        train_sft_baseline(quant_model_dir, dataset, output_dir)
    else:
        print(f"[ERROR] Phương pháp RL không hỗ trợ hoặc sai cấu hình: {method}")
        return
    print(f"\n[SUCCESS] Hoàn tất quá trình huấn luyện {method}! Model được lưu tại: {output_dir}")

def main():
    print("==========================================")
    print(" BẮT ĐẦU PIPELINE END-TO-END QUANT & RL")
    print(f" MODEL: {BASE_MODEL_ID}")
    print(f" RL METHOD: {RL_METHOD}")
    print("==========================================\n")
    
    setup_environment()
    dataset = download_data()
    dataset_path = "./data/science_qa/validation-00000-of-00001-6c7328ff6c84284c.parquet"
    
    base_model_dir = download_model(BASE_MODEL_ID)
    quant_model_dir = run_quantization(base_model_dir, dataset_path, QUANT_BITS)
    
    run_rl_training(quant_model_dir, dataset, RL_METHOD)

if __name__ == "__main__":
    main()