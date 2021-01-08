import numpy as np
from time import time
import matplotlib.pyplot as plt
from matplotlib import rcParams
from cycler import cycler   

class kmeans:
    def __init__(self, n_clusters=2, max_iter=300, verbose=0, color_palet = ['r','g','b']):
        if(n_clusters < 2):
            raise SystemError("Number of clusters (%d) to small." %n_clusters)
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.verbose = verbose
        self.color_palet = ['r','g','b', 'c', 'm', 'y', 'k']
        self.min_step = 0.001
        

    def initialization(self, x):
        ## step 0: initialize centroids
        self.dimensions = x.shape[1]
        self.centroids = np.zeros((self.n_clusters,self.dimensions))
        self.pripadnost = np.zeros(len(x))
        self.most_outer_values = np.array([np.min(x, 0), np.max(x, 0)])

        for centroid in self.centroids:
            for j in range(len(centroid)):
                centroid[j] = np.random.uniform(self.most_outer_values[0, j], self.most_outer_values[1, j])
        if(self.verbose > 1):
            print("Initial centroids are:\n", self.centroids)

    def fit(self, x):
        # step 0
        self.initialization(x)

        ## step 1: find nearest cluster for each point
        # 1. calculate distance from each cluster
        #distances = np.zeros(shape=(len(x), self.n_clusters))

        for it in range(self.max_iter):
            start=time()
            distances = self.getDistancesFromCentroids(x)
            self.pripadnost = np.argmin(distances, axis=1)

            step_lengths = np.zeros(self.n_clusters)

            for i in range(self.n_clusters):
                mask = np.where(self.pripadnost == i)
                new_centroid = np.mean(x[mask], axis=0)
                #print("cluster=%d" %i, x[mask])
                step_lengths[i] = np.linalg.norm(self.centroids[i] - new_centroid)
                self.centroids[i] = new_centroid
                
                if(self.verbose > 1):
                    print("New centroid %d, step length = %f" %(i, step_lengths[i]), self.centroids[i])
                
            if(self.verbose > 1):
                print("Processed finished in %fs" %(time()-start))

            if(np.max(step_lengths) < self.min_step):
                if(self.verbose > 0): 
                    print("Reached minimal step.")
                break
            
        return self.centroids

    def predict(self, x):

        if isinstance(x, np.ndarray) is False:
            x = np.array(x)
        else:
            if x.shape[1] != self.dimensions:
                print("ERROR Input shape", x.shape[1], " expected", self.dimensions)
                return 0
       
        if x.ndim == 1:
            x = x.reshape(-1,2)

            
        print(x)
        distances = np.zeros(shape=(len(x), self.n_clusters))
        
        for i,centroid in enumerate(self.centroids):
            dist = (x - centroid)**2
            dist = np.sum(dist, axis=1)
            dist = np.sqrt(dist)

            distances[:, i] = dist

        return np.argmin(distances, axis=1)

    def getDistancesFromCentroids(self, x):
        distances = np.zeros(shape=(len(x), self.n_clusters))
        for i,centroid in enumerate(self.centroids):
            dist = (x - centroid)**2
            dist = np.sum(dist, axis=1)
            dist = np.sqrt(dist)
            distances[:, i] = dist
        return distances

    def plot(self, x, dim1=0, dim2=1, show_centroid=True):
        rcParams['axes.prop_cycle'] = cycler(color='bgrcmyk')
    
        plt.figure()
        plt.xlabel(r"x$_%d$" %dim1)
        plt.ylabel(r"x$_%d$" %dim2)
        plt.title("Data: gauss")

        if(show_centroid == False):
            plt.scatter(x[:,dim1], x[:, dim2])
        else:
            distances = self.getDistancesFromCentroids(x)
            pripadnost = np.argmin(distances, axis=1)

            for i in range(self.n_clusters):
                mask = np.where(pripadnost == i)
                plt.scatter(x[mask,dim1], x[mask, dim2])
                plt.scatter(self.centroids[i, dim1], self.centroids[i, dim2], c='k', marker='x', s=100)
            