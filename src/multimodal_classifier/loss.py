import tensorflow as tf

class CustomLoss:
    def __init__(self, loss_type='categorical_crossentropy', alpha=0.25, gamma=2, margin=1.0, class_weight=None):
        """
        loss_type: Type of loss function, options include 'categorical_crossentropy', 'categorical_focal', 'contrastive', or 'triplet'.
        alpha: Weighting factor for underrepresented classes in Focal Loss.
        gamma: Focusing parameter in Focal Loss to adjust for easy vs hard samples.
        margin: Margin parameter in Contrastive Loss.
        class_weight: Class weight to handle class imbalance.
        """
        self.loss_type = loss_type
        self.alpha = alpha
        self.gamma = gamma
        self.margin = margin
        self.class_weight = class_weight
        self.loss_fn = self.get_loss_fn(loss_type)

    def get_loss_fn(self, loss_type):
        if loss_type == 'categorical_crossentropy':
            return tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        elif loss_type == 'binary_crossentropy':
            return tf.keras.losses.BinaryCrossentropy(from_logits=False)
        elif loss_type == 'categorical_focal':
            return tf.keras.losses.CategoricalFocalCrossentropy(alpha=self.alpha, gamma=self.gamma, from_logits=False)
        elif loss_type == 'binary_focal':
            return tf.keras.losses.BinaryFocalCrossentropy(alpha=self.alpha, gamma=self.gamma, from_logits=False)
        elif loss_type == 'contrastive':
            return self.contrastive_loss
        elif loss_type == "triplet":
            return self.triplet_loss
        else:
            raise ValueError(f"Invalid loss type: {loss_type}")

    def contrastive_loss(self, y_true, y_pred):
        """
        Contrastive Loss function: y_true = 0 for similar pairs, y_true = 1 for dissimilar pairs.
        """
        # Calculate L2 distance between vectors
        squared_predictions = tf.square(y_pred)
        distances = tf.reduce_sum(squared_predictions, axis=-1)
        
        # Apply Contrastive Loss formula
        loss = y_true * tf.square(tf.maximum(self.margin - distances, 0)) + (1 - y_true) * distances
        return tf.reduce_mean(loss)
        
    def compute_loss(self, y_true, y_pred):
        """
        Compute loss based on selected loss function and apply class_weight if provided.
        """
        loss = self.loss_fn(y_true, y_pred)
        
        if self.class_weight is not None:
            weights = tf.reduce_sum(self.class_weight * y_true, axis=-1)
            weights = tf.expand_dims(weights, axis=-1)  
            loss = loss * weights
        
        return tf.reduce_mean(loss)
