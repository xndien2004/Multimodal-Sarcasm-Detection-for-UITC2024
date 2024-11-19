from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
from PIL import Image
import torch
import pandas as pd
from tqdm import tqdm
import logging
import os

logging.basicConfig(level=logging.INFO)
logging.getLogger("transformers").setLevel(logging.ERROR)

def initialize_models(device):
    logging.info("Loading models...")
    
    model_id = 'microsoft/Florence-2-base'
    vision_model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype='auto').eval().cuda()
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    
    model_name = "vinai/vinai-translate-en2vi-v2"
    tokenizer_en2vi = AutoTokenizer.from_pretrained(model_name, src_lang="en_XX")
    translation_model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    
    logging.info("Models loaded successfully.")
    return vision_model, processor, translation_model, tokenizer_en2vi

def generate_caption(task_prompt, image, vision_model, processor):
    try:
        inputs = processor(text=task_prompt, images=image, return_tensors="pt").to('cuda', torch.float16)
        generated_ids = vision_model.generate(
            input_ids=inputs["input_ids"].cuda(),
            pixel_values=inputs["pixel_values"].cuda(),
            max_new_tokens=1024,
            early_stopping=True,
            do_sample=False,
            num_beams=3,
        )
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(image.width, image.height)
        )
        return parsed_answer[task_prompt]
    except Exception as e:
        logging.error(f"Error in generating caption: {e}")
        return ""

# Hàm dịch tiếng Anh sang tiếng Việt
def translate_en2vi(en_text, translation_model, tokenizer_en2vi, device):
    try:
        input_ids = tokenizer_en2vi(en_text, padding=True, return_tensors="pt").to(device)
        output_ids = translation_model.generate(
            **input_ids,
            decoder_start_token_id=tokenizer_en2vi.lang_code_to_id["vi_VN"],
            num_return_sequences=1,
            num_beams=5,
            early_stopping=True
        )
        return tokenizer_en2vi.batch_decode(output_ids, skip_special_tokens=True)[0]
    except Exception as e:
        logging.error(f"Error in translation: {e}")
        return en_text

# Hàm chính để xử lý dữ liệu
def process_data(data_path, image_folder, output_path, vision_model, processor, translation_model, tokenizer_en2vi, device):
    logging.info("Processing data...")
    
    data = pd.read_json(data_path).T
    task_prompt = '<MORE_DETAILED_CAPTION>'
    
    processed_data = []
    for i in tqdm(data.index):
        try:
            row = data.loc[i]
            image_path = os.path.join(image_folder, row['image'])
            img = Image.open(image_path).convert("RGB")
            
            # Tạo caption hình ảnh
            caption_image = generate_caption(task_prompt, img, vision_model, processor)
            caption_image_vi = translate_en2vi(caption_image, translation_model, tokenizer_en2vi, device)
            
            processed_data.append({
                "image": row['image'],
                "caption": row['caption'],
                "label": row['label'],
                "caption_image": caption_image_vi
            })
        except Exception as e:
            logging.warning(f"Skipping row {i} due to error: {e}")
    
    # Lưu kết quả
    processed_df = pd.DataFrame(processed_data)
    processed_df.to_csv(output_path, index=False)
    logging.info(f"Data processing complete. Results saved to {output_path}")

if __name__ == "__main__":
    data_path = "/kaggle/input/vimmsd/vimmsd-private-test.json"
    image_folder = "/kaggle/input/vimmsd/test-images/"
    output_path = "vimmsd-private-test-new.csv"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    vision_model, processor, translation_model, tokenizer_en2vi = initialize_models(device)
    
    process_data(data_path, image_folder, output_path, vision_model, processor, translation_model, tokenizer_en2vi, device)
