from GD import build_generator,build_discriminator,build_gan,generate_synthetic_data
from Preprocessing import data_fraud
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from train import generator

# generate data
synthetic_data = generate_synthetic_data(generator,500)
df = pd.DataFrame(synthetic_data)
df['label'] = 'fake'
df2 = data_fraud.drop("Class",axis=1)
df2['label'] = 'real'
df2.columns = df.columns
combined = pd.concat([df,df2])

# only show 5 columns, if you want to show all change the code
counter = 0
for col in combined.columns:
    if counter < 5:
        plt.figure()
        fig = px.histogram(combined, color='label', x=col, barmode="overlay", title=f'Feature {col}', width=640, height=500)
        fig.show()
        counter += 1
    else:
        break