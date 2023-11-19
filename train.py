from GD import build_generator,build_discriminator,build_gan,generate_synthetic_data
from Preprocessing import data_fraud
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
generator = build_generator()
discriminator = build_discriminator()
gan = build_gan(generator,discriminator)
gan.compile(optimizer = 'adam', loss = 'binary_crossentropy')

num_epochs = 100
batch_size=64
half = int(batch_size/2)

for epoch in range(num_epochs):
  X_fake = generate_synthetic_data(generator,half)
  y_fake = np.zeros((half,1))

  X_real = data_fraud.drop("Class",axis=1).sample(half)
  y_real = np.ones((half,1))

  discriminator.trainable = True
  discriminator.train_on_batch(X_real,y_real)
  discriminator.train_on_batch(X_fake,y_fake)

  noise = np.random.normal(0,1, (batch_size,29))
  gan.train_on_batch(noise,np.ones((batch_size,1)))

