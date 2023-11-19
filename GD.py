from keras.models import Model
from keras.layers import Input, Dense, LeakyReLU, BatchNormalization
from keras.optimizers import Adam
from keras import backend as K
import numpy as np
def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)

def gradient_penalty_loss(y_true, y_pred, averaged_samples):
    gradients = K.gradients(y_pred, averaged_samples)[0]
    gradients_sqr = K.square(gradients)
    gradient_penalty = K.mean(gradients_sqr)
    return gradient_penalty

def build_discriminator(input_dim=29):
    input_data = Input(shape=(input_dim,))
    x = Dense(128)(input_data)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dense(64)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dense(32)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dense(1)(x)
    model = Model(inputs=input_data, outputs=x)
    model.compile(optimizer=Adam(0.0002, 0.5), loss=wasserstein_loss)
    return model

def build_generator(input_dim=29):
    input_data = Input(shape=(input_dim,))
    x = Dense(32)(input_data)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Dense(64)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Dense(128)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Dense(input_dim, activation='tanh')(x)
    model = Model(inputs=input_data, outputs=x)
    return model
def build_gan(gen,dis):
  dis.trainable = False
  gan_input = Input(shape = (gen.input_shape[1],))

  x = gen(gan_input)

  gan_output = dis(x)

  gan = Model(gan_input,gan_output)
  gan.summary()
  return gan

def generate_synthetic_data(gen,num):
  noise = np.random.normal(0,1,(num,gen.input_shape[1]))
  fake = gen.predict(noise)
  return fake
