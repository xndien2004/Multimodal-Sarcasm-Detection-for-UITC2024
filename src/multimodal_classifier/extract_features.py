import cv2
import numpy as np
import torch
from tqdm import tqdm


class FeatureExtractor:
    def __init__(self, processor, text_model, image_model, tokenizer, device):
        self.device = device
        self.processor = processor
        self.text_model = text_model
        self.image_model = image_model
        self.tokenizer = tokenizer

    def _load_image(self, image_path):
        """Helper function to load an image from a specified path."""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Image not found at path: {image_path}")
        return image

    def _process_image(self, image):
        """Helper function to process and prepare image tensors for feature extraction."""
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.image_model(**inputs)
        return outputs.logits.cpu().numpy().squeeze()

    def extract_image_features(self, images, path_prefix=""):
        """Extract features from a list of images."""
        image_features = []
        for image_name in tqdm(images, desc="Extracting image features"):
            image_path = path_prefix + image_name
            try:
                image = self._load_image(image_path)
                features = self._process_image(image)
                image_features.append(features)
            except Exception as e:
                print(f"Error processing image '{image_name}': {str(e)}")
                image_features.append(np.zeros(1000)) 
        return np.array(image_features)

    def _tokenize_text(self, text):
        """Helper function to tokenize and encode text for feature extraction."""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.text_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).cpu().numpy().squeeze()

    def extract_text_features(self, texts):
        """Extract features from a list of text inputs."""
        text_features = []
        for text in tqdm(texts, desc="Extracting text features"):
            try:
                features = self._tokenize_text(text)
                text_features.append(features)
            except Exception as e:
                print(f"Error processing text: {str(e)}")
                text_features.append(np.zeros(1024))  
        return np.array(text_features)

    def extract_features(self, texts, caption_texts, images, path_prefix=""):
        """
        Extract both text and image features.
        
        Parameters:
        - texts: List of primary text inputs.
        - caption_texts: List of texts used as captions for images.
        - images: List of image file names.
        
        Returns:
        - Tuple of arrays (text_features, image_features, caption_text_features).
        """
        image_features = self.extract_image_features(images, path_prefix=path_prefix)
        text_features = self.extract_text_features(texts)
        caption_text_features = self.extract_text_features(caption_texts)
        
        return text_features, image_features, caption_text_features