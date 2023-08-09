import tensorflow as tf
from tensorflow.keras.models import Model
import tensorflow_addons as tfa
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, ConvLSTM2D, Reshape, Concatenate

def build_encoder(input_shape):
    input_layer = Input(shape=input_shape)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same',strides=1)(input_layer)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', strides=2)(conv1)
    conv3 = Conv2D(128, (2, 2), activation='relu', padding='same', strides=2)(conv2)
    conv4 = Conv2D(256, (2, 2), activation='relu', padding='same', strides=2)(conv3)
    return Model(inputs=input_layer, outputs=conv4)

# def build_attention_gru(input_shape, hidden_units):
#     input_layer = Input(shape=input_shape)
#     attention_gru1 = ConvLSTM2D(hidden_units[0], (3, 3), padding='same', return_sequences=True)(input_layer)
#     attention_gru2 = ConvLSTM2D(hidden_units[1], (3, 3), padding='same', return_sequences=True)(attention_gru1)
#     attention_gru3 = ConvLSTM2D(hidden_units[2], (3, 3), padding='same', return_sequences=True)(attention_gru2)
#     attention_gru4 = ConvLSTM2D(hidden_units[3], (3, 3), padding='same', return_sequences=False)(attention_gru3)
#     return Model(inputs=input_layer, outputs=attention_gru4)

def build_attention_gru(input_shape, attention_units):
    input_layer = Input(shape=input_shape)
    conv_gru1 = tfa.layers.ConvGRU2D(attention_units, (3, 3), padding='same', return_sequences=True)(input_layer)
    conv_gru2 = tfa.layers.ConvGRU2D(attention_units, (3, 3), padding='same', return_sequences=True)(conv_gru1)
    conv_gru3 = tfa.layers.ConvGRU2D(attention_units, (3, 3), padding='same', return_sequences=True)(conv_gru2)
    conv_gru4 = tfa.layers.ConvGRU2D(attention_units, (3, 3), padding='same', return_sequences=True)(conv_gru3)
    return Model(inputs=input_layer, outputs=conv_gru4)

def build_decoder(input_shape):
    input_layer = Input(shape=input_shape)
    conv_transpose1 = Conv2DTranspose(128, (2, 2), activation='relu', padding='same', strides=2)(input_layer)
    conv_transpose2 = Conv2DTranspose(64, (2, 2), activation='relu', padding='same', strides=2)(conv_transpose1)
    conv_transpose3 = Conv2DTranspose(32, (3, 3), activation='relu', padding='same', strides=2)(conv_transpose2)
    output_layer = Conv2D(input_shape[-1], (3, 3), activation='sigmoid', padding='same')(conv_transpose3)
    return Model(inputs=input_layer, outputs=output_layer)

def build_autoencoder(input_shape, attention_units):
    input_layer = Input(shape=input_shape)
    encoder = build_encoder(input_shape)(input_layer)
    attention_gru = build_attention_gru(encoder.shape[1:], attention_units)(encoder)
    decoder = build_decoder(attention_gru.shape[1:])(attention_gru)
    return Model(inputs=input_layer, outputs=decoder)