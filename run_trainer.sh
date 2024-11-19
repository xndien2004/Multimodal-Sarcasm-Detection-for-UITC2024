
#!/bin/bash

pip install -r requirements.txt
SCRIPT_PATH="src/multimodal_classifier/main.py"

# Setting path
TRAIN_PATH="/kaggle/input/vimmsd/train-images/"
TEST_PATH="/kaggle/input/vimmsd/test-images/"
TRAIN_JSON_PATH="/kaggle/input/vimmsd/vimmsd-train-new-translate.csv"
TEST_JSON_PATH="/kaggle/input/vimmsd/vimmsd-private-test-new-translate.csv"
SAVE_MODEL_PATH="/kaggle/working/sarcasm_model.h5"
NAME_TEST_FILE="/kaggle/working/test.json"
IMAGE_MODEL="google/vit-base-patch16-384"
TEXT_MODEL="jinaai/jina-embeddings-v3"

# Run script
python $SCRIPT_PATH \
    --train_path $TRAIN_PATH \
    --test_path $TEST_PATH \
    --train_json_path $TRAIN_JSON_PATH \
    --test_json_path $TEST_JSON_PATH \
    --save_path $SAVE_MODEL_PATH \
    --name_test_file $NAME_TEST_FILE \
    --image_model $IMAGE_MODEL \
    --text_model $TEXT_MODEL \
    --epochs 25 \
    --batch_size 16 \
    --initial_lr 7e-5 \
    --n_classes 4 \
    --class_weight 0.1 0.3 0.5 0.1 \
    --loss_type "categorical_focal" \
    --alpha 0.25 \
    --gamma 2
