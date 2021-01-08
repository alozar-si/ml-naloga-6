# Naloga 1: Implementacija k-means
# V tej skripti je prikazano delovanje k-means, na koncu pa se uporaba z razredom
#%%
import matplotlib.pyplot as plt
import numpy as np
from time import time
# %%
path_to_data = "C:/Users/andlo/data/psuf/ml-naloga-6/podatki1/"
data = np.load(path_to_data + "gauss.npy")
print("Data len = %d" %len(data))
# %%
# Show data on graph
plt.figure(1)
plt.scatter(data[:,0], data[:, 1])
plt.xlabel("x")
plt.ylabel("y")
plt.title("Data: gauss")
# %%
verbose = 2
n_clusters = 2
min_step = 0.001
color_palet = ['r','g','b']
#%%

## step 0: initialize centroids
dimensions = data.ndim
centroids = np.zeros((n_clusters,dimensions))
pripadnost = np.zeros(len(data))
most_outer_values = np.array([np.min(data, 0), np.max(data, 0)])

for centroid in centroids:
    for j in range(len(centroid)):
        centroid[j] = np.random.uniform(most_outer_values[0, j], most_outer_values[1, j])
if(verbose > 0):
    print("Initial centroids are:\n", centroids)

# %%

## step 1: find nearest cluster for each point
# samo za testiranje hitrosti data = np.random.uniform(size=(100000,2))
# 1. calculate distance from each cluster
# 1. metoda
distances = np.zeros(shape=(len(data), n_clusters))
start=time()
for i,centroid in enumerate(centroids):
    dist = (data - centroid)**2
    dist = np.sum(dist, axis=1)
    dist = np.sqrt(dist)

    #dist = np.linalg.norm(data - centroid, axis=1) # ali pa to

    distances[:, i] = dist

pripadnost = np.argmin(distances, axis=1)

print("Processed finished in %fs" %(time()-start))

# 2. metoda, pocasna
'''
distances = []
start=time()
for point in data:
    for i, centroid in enumerate(centroids):
        tmp = np.linalg.norm(point - centroid)
        if i == 0:
            dist = tmp
            closes_centroid=i
        else:
            if(tmp < dist):
                # found closer centroid
                closes_centroid=i
                dist=tmp
         
dist = np.linalg.norm(data - centroid, axis=1)
distances.append(dist)
print("Processed finished in %fs" %(time()-start))
'''
## step 2: calculate position of new centroids

if(verbose > 1):
    plt.figure(1)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Data: gauss")

step_lengths = np.zeros(n_clusters)

for i in range(n_clusters):
    mask = np.where(pripadnost == i)
    new_centroid = np.mean(data[mask], axis=0)

    step_lengths[i] = np.linalg.norm(centroids[i] - new_centroid)
    centroids[i] = new_centroid
    
    if(verbose > 0):
        print("New centroid %d, step length = %f" %(i, step_lengths[i]), centroids[i])
    if(verbose > 1):
        plt.scatter(data[mask,0], data[mask, 1], c=color_palet[i])
        plt.scatter(centroids[i, 0], centroids[i, 1], c='k', marker='x', s=100)

if(np.max(step_lengths) < min_step):
    print("Reached minimal step.")
# %%

## Preizkus razreda kmeans:
from models import kmeans

model = kmeans(n_clusters=2, verbose=4)
model.fit(data)

# %%
x  = np.array([[0,0],[1,1]])
model.predict(x)
# %%
