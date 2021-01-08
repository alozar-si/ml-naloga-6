# Naloga 1: Implementacija k-means 
#%%
import matplotlib.pyplot as plt
import numpy as np
from time import time
# %%
path_to_data = "C:/Users/andlo/data/psuf/ml-naloga-6/podatki1/"
data = np.load(path_to_data + "gauss.npy")
print("Data len = %d" %len(data))
# %%

## Preizkus razreda kmeans:
from models import kmeans

model = kmeans(n_clusters=2, verbose=4)
model.fit(data)

# %%
x  = np.array([[0,0],[1,1]])
model.predict(x)
model.plot(x)
# %%
