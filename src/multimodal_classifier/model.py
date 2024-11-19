from tensorflow.keras.layers import Input, Dense, Dropout, concatenate, LayerNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import MultiHeadAttention


class SarcasmModel:
    def __init__(self, config):
        self.config = config
        self.model = self.build_model()

    def build_model(self):
        # Define inputs
        image_input = Input(shape=(self.config.image_dim,), name='image_input')
        text_input = Input(shape=(self.config.text_dim,), name='text_input')
        caption_input = Input(shape=(self.config.caption_image_dim,), name='caption_input')
        
        # Create branches
        image_branch = self._create_branch(image_input, self.config.image_branch_layers)
        text_branch = self._create_branch(text_input, self.config.text_branch_layers)
        caption_branch = self._create_branch(caption_input, self.config.caption_image_branch_layers)
        
        # # Multi-head attention (text - image)
        # image_text_attention = self._multihead_attention(caption_branch, image_branch)
        
        # # Co-attention (text - caption)
        # text_caption_attention = self._coattention(text_branch, caption_branch)

        # Combine all features 
        combined = concatenate([image_branch, text_branch, caption_branch])
        combined = Dense(self.config.combined_layer_size, activation='relu')(combined)
        combined = Dropout(self.config.dropout_rate)(combined)
        combined = Dense(int(self.config.combined_layer_size/2), activation='relu')(combined)
        combined = Dropout(self.config.dropout_rate)(combined) 
        
        # Output layer
        output = Dense(self.config.n_classes, activation='softmax', name="output")(combined)
        
        return Model(inputs=[image_input, text_input, caption_input], outputs=output)

    def _create_branch(self, input_layer, layer_sizes):
        x = input_layer
        for size in layer_sizes:
            x = Dense(size, activation='relu')(x)
            x = Dropout(self.config.dropout_rate)(x)
        return x

    def _multihead_attention(self, text_branch, image_branch):
        # Add Multi-head Attention between text and image branches
        text_branch = Dense(self.config.key_dim)(text_branch)
        image_branch = Dense(self.config.key_dim)(image_branch)

        text_attention = layers.Reshape((-1, self.config.key_dim))(text_branch)
        image_attention = layers.Reshape((-1, self.config.key_dim))(image_branch)
        
        # Apply multi-headed attention
        attention_output = MultiHeadAttention(num_heads=self.config.num_heads, key_dim=self.config.key_dim)(text_attention, image_attention)
        attention_output = LayerNormalization()(attention_output)
        attention_output = layers.Reshape((self.config.key_dim,))(attention_output)
        return attention_output

    def _coattention(self, text_branch, caption_branch):
        # Co-attention mechanism between text and caption
        text_branch = Dense(self.config.key_dim)(text_branch)
        caption_branch = Dense(self.config.key_dim)(caption_branch)

        text_attention = layers.Reshape((-1, self.config.key_dim))(text_branch)
        caption_attention = layers.Reshape((-1, self.config.key_dim))(caption_branch)
        text_caption_attention = MultiHeadAttention(num_heads=self.config.num_heads, key_dim=self.config.key_dim)(text_attention, caption_attention)
        text_caption_attention = LayerNormalization()(text_caption_attention)
        text_caption_attention = layers.Reshape((self.config.key_dim,))(text_caption_attention)
        return text_caption_attention