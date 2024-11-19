import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from loss import CustomLoss
from callback import F1Callback, LossPlotCallback
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class Trainer:
    def __init__(self, config, model, evaluator):
        self.model = model
        self.config = config
        self.evaluator = evaluator
        self.train_losses = [] 
        self.val_losses = []   

    def encode_labels(self, labels):
        if not hasattr(self.config, 'map_label') or not hasattr(self.config, 'n_classes'):
            raise ValueError("Config must have `map_label` and `n_classes` attributes.")
        return tf.keras.utils.to_categorical([self.config.map_label[label] for label in labels], num_classes=self.config.n_classes)

    def decode_labels(self, one_hot_labels):
        reverse_mapping = {v: k for k, v in self.config.map_label.items()}
        print(reverse_mapping)
        return [reverse_mapping[idx] for idx in np.argmax(one_hot_labels, axis=1)]

    def train(self, train_images, train_texts, train_captions, train_labels,  val_images, val_texts, val_captions, val_labels):
        train_labels = self.encode_labels(train_labels)
        val_labels = self.encode_labels(val_labels)
        # Define custom loss
        loss_fn = CustomLoss(self.config.loss_type, self.config.alpha, self.config.gamma, 
                             class_weight=tf.constant(self.config.class_weight), margin=self.config.margin)
        
        # Compile model
        self.model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=self.config.initial_lr),
                           loss=loss_fn.compute_loss, metrics=[tf.keras.metrics.AUC()])
        # tf.config.optimizer.set_experimental_options({"xla": False})

        # Define callbacks
        callbacks = [
            F1Callback(val_images, val_texts, val_captions, val_labels), 
            tf.keras.callbacks.LearningRateScheduler(self._adjust_learning_rate),
            LossPlotCallback(self)
        ]
        
        # Train model
        self.model.fit([train_images, train_texts, train_captions], train_labels,
                       validation_data=([val_images, val_texts, val_captions], val_labels),
                       epochs=self.config.epochs, batch_size=self.config.batch_size,
                       callbacks=callbacks)

        # plot loss
        self.plot_loss()
        # Evaluate model
        self.evaluator.evaluate(self.model, val_images, val_texts, val_captions, val_labels)

    def _adjust_learning_rate(self, epoch, lr):
        if epoch >= self.config.lr_schedule['decay_mid']:
            return lr * self.config.lr_schedule['decay_factor_end']
        elif epoch >= self.config.lr_schedule['decay_start']:
            return lr * self.config.lr_schedule['decay_factor_mid']
        return lr

    def plot_loss(self):
        """Plot training and validation loss."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
    
    def predict(self, images, texts, captions):
        predictions = self.model.predict([images, texts, captions])
        list_prods = predictions.tolist()
        return self.decode_labels(predictions), list_prods

    def save_model(self, path):
        self.model.save(path)
        print(f"Model saved at {path}")

    def load_model(self, path):
        self.model = tf.keras.models.load_model(path, compile=False)
        print(f"Model loaded from {path}")

class Evaluator:
    def __init__(self, config):
        self.config = config

    def evaluate(self, model, val_images, val_texts, val_captions, val_labels): 
        predictions = model.predict([val_images, val_texts, val_captions])
        y_pred_classes = np.argmax(predictions, axis=1)
        y_true_classes = np.argmax(val_labels, axis=1)

        auc = tf.keras.metrics.AUC()(val_labels, predictions)
        print(f"AUC: {auc.numpy()}")

        self.plot_confusion_matrix(y_true_classes, y_pred_classes)

    def plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred, labels=list(self.config.map_label.values()))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(self.config.map_label.keys()))
        
        fig, ax = plt.subplots(figsize=(8, 8))
        disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='d')
        plt.title("Confusion Matrix")
        plt.show()
