import sys
import os
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, GPTQConfig, AutoConfig

# Đảm bảo load được module dữ liệu
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.dataset_loader import ScienceQALocalLoader

class QwenGPTQQuantizer:
    def __init__(self, base_model_path, save_path, data_path):
        self.base_model_path = base_model_path
        self.save_path = save_path
        self.data_path = data_path

    def get_calibration_data(self, test_size=8):
        print(f"--- Đang tải dữ liệu calibration (Size: {test_size}) ---")
        loader = ScienceQALocalLoader(self.data_path, subset_size=test_size)
        df = loader.preprocess_for_r3_quant()
        # Doc: "You could also pass your own dataset as a list of strings"
        return [f"Question: {row['question']}\nAnswer: {row['reasoning']}" for _, row in df.iterrows()]

    def quantize_and_save(self, bits=3):
        calib_dataset = self.get_calibration_data(test_size=8)
        
        # SỬA LỖI TẠI ĐÂY: Sử dụng GPTQConfig theo đúng tài liệu HF bạn gửi
        gptq_config = GPTQConfig(
            bits=bits,
            dataset=calib_dataset,
            tokenizer=self.base_model_path, # Doc: cần tokenizer để prep dataset
            use_exllama=False,             # 3-bit không hỗ trợ ExLlama
            desc_act=False,
            sym=True
        )

        # Vá lỗi use_cache đặc thù của dòng Qwen2.5-VL
        config = AutoConfig.from_pretrained(self.base_model_path)
        config.use_cache = False

        print(f"--- Đang tải model và bắt đầu lượng tử hóa ({bits}-bit)... ---")
        try:
            # Load model theo đúng hướng dẫn trong Doc (from_pretrained + quantization_config)
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.base_model_path,
                config=config,
                quantization_config=gptq_config,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True
            )

            # Lưu model (Theo Doc: nên đưa về CPU nếu dùng device_map)
            print("--- Lượng tử hóa xong. Đang lưu model... ---")
            model.to("cpu")
            os.makedirs(self.save_path, exist_ok=True)
            model.save_pretrained(self.save_path)
            
            processor = AutoProcessor.from_pretrained(self.base_model_path)
            processor.save_pretrained(self.save_path)
            print(f"--- Hoàn tất! Model đã lưu tại: {self.save_path} ---")
            
        except Exception as e:
            print(f"--- Lỗi trong quá trình xử lý: {e} ---")
            # THÊM DÒNG NÀY: Để file main.py không báo [SUCCESS] giả
            sys.exit(1) 

if __name__ == "__main__":
    BASE_MODEL = r"./weights/Qwen2.5-VL-3B-Instruct"
    SAVE_DIR = r"./weights/Qwen2.5-VL-3B-Instruct-GPTQ-Int3"
    DATA_PATH = r"./data/science_qa/validation-00000-of-00001-6c7328ff6c84284c.parquet"
    
    quantizer = QwenGPTQQuantizer(BASE_MODEL, SAVE_DIR, DATA_PATH)
    quantizer.quantize_and_save(bits=3)