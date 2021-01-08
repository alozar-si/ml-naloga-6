# Naloga 2: Uporaba implementacije k-means na podatkih h_bb.npy
#%%
import matplotlib.pyplot as plt
import numpy as np
from time import time
from models import kmeans
from utils import calc_p, n_clusters_vs_mass, n_clusters_vs_mass_scikit
from sklearn.cluster import KMeans

path_to_data = "C:/Users/andlo/data/psuf/ml-naloga-6/podatki1/"
data = np.load(path_to_data + "h_bb.npy", allow_pickle=True)
print("Število dogodkov = %d" %len(data))
print("Števil trackov v prvem dogodku = %d" %(len(data[0])))

## Naucimo kmeans na prvem dogodku:
x = np.array(data[0])[:, 0:3]
#%%
# Uporaba scikit-learn
model = KMeans(n_clusters=10, verbose=0).fit(x)
results = model.cluster_centers_

#%%
# Zveza med st. clustrov in izracunano masso
masses = n_clusters_vs_mass(x, max_clusters=10, repeat=10)

#%%
# Zveza med st. clustron in izracunano maso - scikit-learn
[masses, mass_std] = n_clusters_vs_mass_scikit(x, max_clusters=10, repeat=10)
#%%
# k = 2
model2 = kmeans(n_clusters=6, verbose=0)
result2 = model2.fit(x)
print(result2)
model2.plot(x, dim1=0, dim2=1, show_centroid=True)

p = calc_p(result2[:, 0], result2[:, 1])
p.sort()
p_jets = np.array([p[-1], p[-1]])

mass = np.sqrt(np.sum(p_jets**2))
print("(p1, p2, ...) = ", p)
print("Higs bosson mass is ", mass)
# %%
# k = 10
model10 = kmeans(n_clusters=10, verbose=0, max_iter=10)
result_10 = model10.fit(x)
print(result_10)
model10.plot(x, dim1=0, dim2=1, show_centroid=False)



# %%
