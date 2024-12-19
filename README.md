<h1 align="center"> 🏆 Multimodal Sarcasm Detection for UITC2024 </h1>

<p align="center">
  <em>A multimodal sarcasm detection system utilizing image-caption generation and natural language processing, developed for the UITC2024 competition, where we achieved 1st place.</em>
</p>

![Static Badge](https://img.shields.io/badge/python->=3.10-blue)
![Static Badge](https://img.shields.io/badge/transformers-4.x-blue)
![Static Badge](https://img.shields.io/badge/sentencepiece-0.2.0-blue)
![Static Badge](https://img.shields.io/badge/pyvi-0.1.1-blue)
![Static Badge](https://img.shields.io/badge/einops-0.8.0-blue)

<details>
  <summary>Table of Contents</summary>

  - [📍 Overview](#-overview)
  - [🎯 Features](#-features)
  - [🏅 Results](#-results)
  - [🚀 Setup and Usage](#-setup-and-usage)
  - [👣 Workflow](#-workflow)
  - [📐 App Structure](#-app-structure)
  - [🧑‍💻 Contributors](#-contributors)

</details>

## 📍 Overview 
The **Multimodal Sarcasm Detection System** is designed to detect sarcasm in multimedia content using image-text pairs. It generates captions from images using a pre-trained Florence-v2 and translate-en2vi model, then processes the data through three input streams: original text, generated captions, and image features. The system integrates these inputs into a unified model that classifies sarcasm across different categories: text sarcasm, image sarcasm, multi-modal sarcasm, and no sarcasm.

This system is developed for the [**UITC2024** competition](https://dsc.uit.edu.vn/bang-b/) and aims to advance the understanding of sarcasm detection in multimodal contexts.

## 🎯 Features

1. **Multimodal Input Handling**
   - **Text-based input**: Handles original text and captions generated from images.
   - **Image-based input**: Generates image captions for context-based analysis.

2. **Sarcasm Classification**
   - Classifies content into four categories: **image sarcasm**, **text sarcasm**, **multi sarcasm**, and **not sarcasm**.

3. **Model Architecture**
   - Utilizes state-of-the-art models like **Florence-v2** for image captioning and **transformers** for text analysis.
   - Integrated **ViT** and **Jina Embedding V3** for feature extraction, with optimization using **Cross Entropy** and **Focal Loss**.

4. **Voting Model Integration**
   - Combines the predictions of four different models trained for 2-class, 3-class, and 4-class tasks to ensure accurate final predictions.

## 🏅 Results

| Team Name      | F1           | Precision     | Recall       |
|----------------|--------------|---------------|--------------|
| **Faster-United** | **0.4475**    | 0.4403        | 0.4563       |
| **US1**           | 0.4403    | 0.4462        | 0.5678       |
| **AIbou**         | 0.4386    | 0.4256        | 0.4935       |
| **BEd**           | 0.4328    | 0.4240        | 0.4574       |
| **MeowProfs**     | 0.4293    | 0.4185        | 0.4511       |

Our team **Faster-United** achieved **1st place** with an F1 score of **0.4475**. The table above shows the top 5 teams and their corresponding F1, Precision, and Recall scores. We are proud of the results and our system's performance across various metrics in the UITC2024 competition.


## 🚀 Setup and Usage

1. **Clone the Repository**
   ```bash
   git clone https://github.com/xndien2004/Multimodal-Sarcasm-Detection-for-UITC2024.git
   cd Multimodal-Sarcasm-Detection-for-UITC2024
   ```

2. **Install Dependencies**
   Make sure Python is installed and then install the necessary dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. **Run trainer**
   To run train, execute:
   ```bash
   bash run_trainer.sh
   ```
   This will start the application and allow you to test the sarcasm detection on your input data.

## 👣 Workflow
<!-- ![Pipeline](./pic/pipeline.png) -->
- **Data Processing**: The system processes image and text data, generating captions for images and using the original text for classification.
- **Model Training**: The four models (trained for 2-class, 3-class, and 4-class tasks) work together to detect sarcasm across different types of input.
- **Voting Model**: The predictions of individual models are aggregated using a Voting Model to produce the final classification.

## 📐 App Structure
```
├── Multimodal-Sarcasm-Detection-for-UITC2024/
│   ├── config/
│   │   ├── config_trainer.yaml
│   ├── pic/
│   ├── src/
│   │   ├── data_processing/
│   │   ├── multimodal_classifier/
│   │   ├── pipeline_notebook/
|   |   ├── utils.py
│   ├── requirements.txt
```

## 🧑‍💻 Contributors

- [Trần Xuân Diện](https://github.com/dienlamAI)
- Võ Trọng Nhơn
- Nguyễn Đăng Tuấn Huy
