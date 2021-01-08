import numpy as np
from models import kmeans
from sklearn.cluster import KMeans

def calc_p(pt, eta):
    return pt * np.cosh(eta)


def n_clusters_vs_mass(x, max_clusters=10, repeat=10):
    masses = []
    for i in range(2, max_clusters+1):
        tmp = []
        for j in range(repeat):
            model = kmeans(n_clusters=i, verbose=0)
            result = model.fit(x)
            p = calc_p(result[:, 0], result[:, 1])
            p.sort()
            p_jets = np.array([p[-1], p[-1]])
            mass = np.sqrt(np.sum(p_jets**2))
            if(np.isnan(mass)==False):
                tmp.append(mass)
        
        if(len(tmp)==0):
            masses.append(0)
        else:
            masses.append(np.mean(tmp))
    print("Finished")
    return masses

def n_clusters_vs_mass_scikit(x, max_clusters=10, repeat=10):
    masses = []
    mass_std = []
    for i in range(2, max_clusters+1):
        print("Clusters: %d" %i)
        tmp = []
        for j in range(repeat):
            model = KMeans(n_clusters=i, verbose=0).fit(x)
            result = model.cluster_centers_
            p = calc_p(result[:, 0], result[:, 1])
            p.sort()
            p_jets = np.array([p[-1], p[-1]])
            mass = np.sqrt(np.sum(p_jets**2))

            if(np.isnan(mass)==False):
                tmp.append(mass)
        
        if(len(tmp)==0):
            masses.append(0)
        else:
            masses.append(np.mean(tmp))
            mass_std.append(np.std(tmp))
    print("Finished")
    return [masses, mass_std]