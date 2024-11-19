import tensorflow as tf
import numpy as np
from sklearn.metrics import f1_score

class F1Callback(tf.keras.callbacks.Callback):
    def __init__(self, val_image, val_text, val_caption_image, val_label):
        super(F1Callback, self).__init__()
        self.val_image = val_image
        self.val_text = val_text
        self.val_label = val_label
        self.val_caption_image = val_caption_image

    def on_epoch_end(self, epoch, logs=None):
        y_true = self.val_label
        y_pred = self.model.predict([self.val_image, self.val_text, self.val_caption_image])
        y_pred = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_true, axis=1)
        f1 = f1_score(y_true, y_pred, average='macro')
        print(f"Validation F1 Score: {f1:.4f}")

class LossPlotCallback(tf.keras.callbacks.Callback):
    def __init__(self, trainer):
        super(LossPlotCallback, self).__init__()
        self.trainer = trainer

    def on_epoch_end(self, epoch, logs=None): 
        self.trainer.train_losses.append(logs.get('loss'))
        self.trainer.val_losses.append(logs.get('val_loss'))