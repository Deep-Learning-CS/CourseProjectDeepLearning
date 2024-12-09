# -*- coding: utf-8 -*-
"""model_development.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1G8IeRAB3PZOgTA6SZPX19AtJER1hwt-0

# Define the Model
"""

from tensorflow import keras

from keras.applications import VGG16, ResNet50

# Load a pretrained model and return the model.
def load_pretrained_model(model_name: str) -> keras.Model:
  if model_name == 'vgg':
    return VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
  elif model_name == 'resnet':
    return ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
  else:
    return None

model = load_pretrained_model('vgg')

# Freeze the layers of a pretrained model to retain learned features.
# Return the model.
def freeze_layers(model: keras.Model) -> keras.Model:
  for layer in model.layers:
    layer.trainable = False
  return model

model = freeze_layers(model)
model.summary()

from keras.layers import Conv2D, UpSampling2D, BatchNormalization
from keras.models import Model

# Add custom layers to the pretrained model to turn it into an
# encoder-decoder architecture for audio denoising
def add_custom_layers(model: keras.Model) -> keras.Model:
  base_model = model

  # set the encoder to the pretrained model output
  encoder = model.output

  # Bottleneck - lower dimensional hidden layer where encoding is produced
  x = Conv2D(512, (3, 3), activation='relu', padding='same')(encoder)
  x = BatchNormalization()(x)

  def decoder_layers(x: keras.Model, unit_size: int) -> keras.Model:
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(unit_size, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    return x

  # Decoder - turn it back into a spectrogram
  x = decoder_layers(x, 256)
  x = decoder_layers(x, 128)
  x = decoder_layers(x, 64)
  x = decoder_layers(x, 32)
  x = decoder_layers(x, 16)

  outputs = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

  return Model(inputs=base_model.input, outputs=outputs)

model = add_custom_layers(model)
model.summary()

# Compile the model using the given optimizer and loss function
def compile_model(model: keras.Model, optimizer: str, loss_function: str):
  model.compile(optimizer=optimizer, loss=loss_function, metrics=['mae'])

compile_model(model, 'rmsprop', 'mean_squared_error')

# Save the model to the given file path
def save_model(model: keras.Model, file_path: str):
  model.save(file_path)

save_model(model, "updated_model.keras")