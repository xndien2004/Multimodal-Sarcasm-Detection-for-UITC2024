import os
import logging

import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, AutoModel

# Setup logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("transformers").setLevel(logging.ERROR)

# Constants
MODEL_ID = '5CD-AI/Vintern-1B-v2'
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Model initialization
def initialize_models(device):
    logging.info("Loading models...")

    model = AutoModel.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        trust_remote_code=True
    ).to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    generation_config = dict(
        max_new_tokens=256,
        do_sample=False,
        num_beams=3,
        repetition_penalty=2.0
    )

    logging.info("Models loaded successfully.")
    return model, tokenizer, generation_config

# Image transformations
def build_transform(input_size):
    return T.Compose([
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

# Image loader
def load_image(image_file, input_size=448, device='cuda'):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size)
    pixel_values = transform(image).unsqueeze(0)
    return pixel_values.to(torch.float16).to(device)

# Caption generation
def generate_caption(image_path, model, tokenizer, generation_config, device, task_prompt=None):
    try:
        pixel_values = load_image(image_path, device=device)

        if task_prompt is None:
            task_prompt = """<image>
            Hãy mô tả nội dung chính của bức ảnh một cách chi tiết và tự nhiên.  
            Bắt đầu bằng một câu mô tả tổng quan về bức ảnh, sau đó tiếp tục miêu tả chủ thể chính, hành động đang diễn ra, bối cảnh xung quanh và cảm xúc mà bức ảnh truyền tải.  
            Nếu trong ảnh có văn bản, hãy trích xuất nội dung chính của nó và mô tả xem văn bản đó đóng vai trò gì trong bối cảnh của bức ảnh.  
            Hãy viết dưới dạng một đoạn văn mạch lạc, không rời rạc, nhưng vẫn đảm bảo đầy đủ thông tin.  
            Chỉ mô tả những gì thực sự có trong ảnh, không suy diễn thêm.
            """

        caption = model.chat(
            tokenizer,
            pixel_values,
            task_prompt,
            generation_config,
            return_history=False
        )
        return caption

    except Exception as e:
        logging.error(f"Error generating caption for {image_path}: {e}")
        return ""

# Data processing pipeline
def process_data(data_path, image_folder, output_path, model, tokenizer, generation_config, device):
    logging.info("Processing data...")

    data = pd.read_json(data_path).T
    processed_data = []

    for idx, row in tqdm(data.iterrows(), total=len(data)):
        image_path = os.path.join(image_folder, row['image'])

        description = generate_caption(
            image_path,
            model,
            tokenizer,
            generation_config,
            device
        )

        processed_data.append({
            "image": row['image'],
            "caption": row['caption'],
            "label": row['label'],
            "caption_image": description
        })

    processed_df = pd.DataFrame(processed_data)
    processed_df.to_csv(output_path, index=False)
    logging.info(f"Data processing complete. Results saved to {output_path}")

# Main execution
if __name__ == "__main__":
    data_path = "/kaggle/input/vimmsd/vimmsd-private-test.json"
    image_folder = "/kaggle/input/vimmsd/test-images/"
    output_path = "vimmsd-private-test-new.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer, generation_config = initialize_models(device)

    process_data(
        data_path,
        image_folder,
        output_path,
        model,
        tokenizer,
        generation_config,
        device
    )