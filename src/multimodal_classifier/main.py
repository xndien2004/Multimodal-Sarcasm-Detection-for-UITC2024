import argparse
import torch
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import logging
import os
import warnings
import seaborn as sns
from transformers import AutoTokenizer, AutoModel, AutoModelForImageClassification, AutoImageProcessor

from data_processing import data_augmentation, processing_text
from extract_features import FeatureExtractor
from model import SarcasmModel
from trainer import Trainer, Evaluator
from utils import CaptionProcessor



warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)


def get_args():
    parser = argparse.ArgumentParser(description="Configuration arguments for the model")

    # Paths
    parser.add_argument("--train_path", type=str, default="/kaggle/input/vimmsd/train-images/", help="Path to training images")
    parser.add_argument("--test_path", type=str, default="/kaggle/input/vimmsd/test-images/", help="Path to testing images")
    parser.add_argument("--train_json_path", type=str, default="/kaggle/input/vimmsd/vimmsd-train-new-translate.csv", help="Path to training JSON file")
    parser.add_argument("--test_json_path", type=str, default="/kaggle/input/vimmsd/vimmsd-private-test-new-translate.csv", help="Path to testing JSON file")

    # Dimensions
    parser.add_argument("--text_model", type=str, default="jinaai/jina-embeddings-v3", help="Text model")
    parser.add_argument("--image_model", type=str, default="google/vit-base-patch16-384", help="Image model")
    parser.add_argument("--image_dim", type=int, default=1000, help="Image feature dimension")
    parser.add_argument("--text_dim", type=int, default=1024, help="Text feature dimension")
    parser.add_argument("--caption_image_dim", type=int, default=1024, help="Caption image feature dimension")

    # Model architecture
    parser.add_argument("--num_heads", type=int, default=4, help="Number of heads in attention layers")
    parser.add_argument("--key_dim", type=int, default=128, help="Key dimension in attention layers")
    parser.add_argument("--image_branch_layers", nargs='+', type=int, default=[1024, 512], help="Layers in image branch")
    parser.add_argument("--text_branch_layers", nargs='+', type=int, default=[512, 256], help="Layers in text branch")
    parser.add_argument("--caption_image_branch_layers", nargs='+', type=int, default=[512, 256], help="Layers in caption-image branch")
    parser.add_argument("--combined_layer_size", type=int, default=1024, help="Size of combined layer")
    parser.add_argument("--dropout_rate", type=float, default=0.3, help="Dropout rate")

    # Training parameters
    parser.add_argument("--epochs", type=int, default=25, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--initial_lr", type=float, default=7e-5, help="Initial learning rate")
    parser.add_argument("--lr_schedule_decay_start", type=int, default=5, help="Learning rate decay start epoch")
    parser.add_argument("--lr_schedule_decay_mid", type=int, default=20, help="Learning rate decay mid epoch")
    parser.add_argument("--lr_schedule_decay_factor_mid", type=float, default=0.1, help="Learning rate decay factor for mid")
    parser.add_argument("--lr_schedule_decay_factor_end", type=float, default=0.01, help="Learning rate decay factor for end")

    # Classification parameters
    parser.add_argument("--map_label", type=dict, default={"not-sarcasm": 0, "image-sarcasm": 1, "text-sarcasm": 2, "multi-sarcasm": 3}, help="Mapping of labels to integers")
    parser.add_argument("--n_classes", type=int, default=4, help="Number of classes")
    parser.add_argument("--class_weight", nargs='+', type=float, default=[0.1, 0.3, 0.5, 0.1], help="Class weights for loss function")
    
    # Loss parameters
    parser.add_argument("--loss_type", type=str, default="categorical_focal", choices=["categorical_crossentropy", "categorical_focal", "binary_crossentropy", "binary_focal", "contrastive", "triplet"], help="Type of loss function")
    parser.add_argument("--alpha", type=float, default=0.25, help="Alpha for focal loss")
    parser.add_argument("--gamma", type=float, default=2, help="Gamma for focal loss")
    parser.add_argument("--margin", type=float, default=1.0, help="Margin for triplet/contrastive loss")

    # load_save_path
    parser.add_argument("--load_model", type=str, default="", help="Path to load model")
    parser.add_argument("--save_path", type=str, default="/kaggle/working/sarcasm_model.h5", help="Path to save model")

    # Testing model
    parser.add_argument("--name_test_file", type=str, default="/kaggle/working/test.json", help="Path to save test file")
    parser.add_argument("--phase", type=str, default="test", help="Phase of testing")

    args = parser.parse_args()
    return args


def main(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    train_df = pd.read_csv(args.train_json_path)

    processing_text.VietnameseTextPreprocessor.initialize_vowel_to_ids()
    train_df['caption'] = train_df['caption'].apply(processing_text.VietnameseTextPreprocessor.unicode_normalize)
    train_df['caption'] = train_df['caption'].apply(processing_text.VietnameseTextPreprocessor.preprocess)
    print(train_df["label"].value_counts())

    # sns countplot
    sns.countplot(x='label', data=train_df)

    # data augmentation
    # upsample_classes = {
    #     'text-sarcasm': 500, 
    #     'image-sarcasm': 1000 
    # }
    train_df = data_augmentation.augment_data(train_df, upsample_classes=None)

    # split data
    train_df, val_df = train_test_split(train_df, test_size=0.2, stratify=train_df["label"], random_state=42)
    print(train_df["label"].value_counts())
    print(val_df["label"].value_counts())

    # load models and tokenizer
    processor = AutoImageProcessor.from_pretrained(args.image_model)
    image_model = AutoModelForImageClassification.from_pretrained(args.image_model).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.text_model)
    text_model = AutoModel.from_pretrained(args.text_model, 
                                                trust_remote_code=True,
                                                torch_dtype=torch.float32).to(device)
    
    # Initialize feature extractor
    feature_extractor = FeatureExtractor(processor, text_model, image_model, tokenizer, device)
    text_train_features, image_train_features, caption_image_train_features = feature_extractor.extract_features(train_df['caption'], train_df['caption_image'], train_df['image'], args.train_path)
    text_val_features, image_val_features, caption_image_val_features = feature_extractor.extract_features(val_df['caption'], val_df['caption_image'], val_df['image'], args.train_path)

    # Initialize model
    model = SarcasmModel(args).model
    evaluator = Evaluator(args)
    trainer = Trainer(args, model, evaluator)
    if args.load_model != "":
        trainer.load_model(args.load_model)
    
    # Train model
    trainer.train(image_train_features, text_train_features, caption_image_train_features, train_df['label'], image_val_features, text_val_features, caption_image_val_features, val_df["label"])

    # Save model
    trainer.save_model(args.save_path)

    # Testing model
    test_df = pd.read_csv(args.test_json_path)
    test_df['caption'] = test_df['caption'].apply(processing_text.VietnameseTextPreprocessor.unicode_normalize)
    test_df['caption'] = test_df['caption'].apply(processing_text.VietnameseTextPreprocessor.preprocess)

    text_test_features, image_test_features, caption_image_test_features = feature_extractor.extract_features(test_df['caption'], test_df['caption_image'], test_df['image'], args.test_path)

    # predict
    predictions, prob = trainer.predict(image_test_features, text_test_features, caption_image_test_features)
    prob = np.max(prob, axis=1).tolist()

    test_df["label"] = predictions
    test_df["prob"] = prob

    # Caption Label Voting by Majority and Probability
    processor = CaptionProcessor(test_df)
    processed_df = processor.process()

    # Save results
    results = {str(i): pred for i, pred in enumerate(list(processed_df["label"]))}
    output = {
        "results": results,
        "phase": args.phase
    }

    with open(args.name_test_file, 'w') as f:
        json.dump(output, f, indent=2)

if __name__ == "__main__":
    args = get_args()
    main(args)

