import re

def extract_xml_answer(text: str) -> str:
    """
    Trích xuất nội dung nằm trong thẻ <answer>...</answer>.
    Nếu không có, trả về chuỗi rỗng.
    """
    answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if answer_match:
        return answer_match.group(1).strip()
    return ""

def format_reward_func(completions, **kwargs) -> list[float]:
    """
    Thưởng 1.0 điểm nếu mô hình tuân thủ đúng định dạng tư duy:
    <think> ... </think> <answer> ... </answer>
    """
    rewards = []
    for comp in completions:
        content = comp[0]["content"] if isinstance(comp, list) else comp
        
        has_think = "<think>" in content and "</think>" in content
        has_answer = "<answer>" in content and "</answer>" in content
        
        if has_think and has_answer:
            rewards.append(1.0)
        else:
            rewards.append(0.0) 
            
    return rewards

def accuracy_reward_func(completions, ground_truth, **kwargs) -> list[float]:
    """
    Thưởng 1.0 điểm nếu đáp án trong thẻ <answer> khớp chính xác với ground_truth.
    """
    rewards = []
    for comp, truth in zip(completions, ground_truth):
        content = comp[0]["content"] if isinstance(comp, list) else comp
        pred_answer = extract_xml_answer(content)
        
        if pred_answer.lower().strip() == truth.lower().strip():
            rewards.append(1.0)
        else:
            rewards.append(0.0)
            
    return rewards

def vision_grounding_reward_func(completions, hint, **kwargs):
    """
    Hàm thưởng Thị giác (Vision-Grounded Reward):
    Mục đích: Đánh giá xem mô hình có thực sự "nhìn" vào ảnh để suy luận không.
    Luật: Moi nội dung trong thẻ <think>. Lấy các từ khóa dài (>4 ký tự) từ cột `hint` (chứa mô tả ảnh).
    Nếu mô hình nhắc đến các từ khóa này trong suy nghĩ -> Thưởng điểm!
    """
    rewards = []
    
    for comp, current_hint in zip(completions, hint):
        text = comp[0]["content"] if isinstance(comp, list) else comp
        
        # Moi nội dung suy nghĩ ra
        match = re.search(r"<think>(.*?)</think>", text, re.DOTALL | re.IGNORECASE)
        
        if match and current_hint.strip() != "":
            think_content = match.group(1).lower()
            
            # Trích xuất các từ khóa (keywords) từ mô tả ảnh (bỏ qua các từ ngắn như is, the, a...)
            keywords = [word.lower() for word in current_hint.split() if len(word) > 4]
            
            if not keywords:
                rewards.append(0.0)
                continue
                
            # Đếm số lượng từ khóa xuất hiện trong suy nghĩ của AI
            hit_count = sum(1 for kw in keywords if kw in think_content)
            hit_ratio = hit_count / len(keywords)
            
            # Tính điểm Reward dựa trên tỷ lệ "trúng" từ khóa
            if hit_ratio >= 0.5:
                rewards.append(1.0) # Nhắc đến hơn 50% chi tiết ảnh -> Thưởng tuyệt đối!
            elif hit_ratio > 0.0:
                rewards.append(0.5) # Nhắc đến 1 phần -> Thưởng 0.5
            else:
                rewards.append(0.0) # Không nhắc đến chi tiết nào trên ảnh -> 0 điểm
        else:
            rewards.append(0.0) # Không có thẻ think hoặc ảnh không có mô tả -> 0 điểm
            
    return rewards