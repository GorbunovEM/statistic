import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import patsy
import numpy as np
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.manifold import TSNE
from sklearn import preprocessing
from scipy.spatial.distance import pdist

iris = sm.datasets.get_rdataset('iris')
iris_data = pd.DataFrame(iris.data)
iris_data.rename(columns={'Sepal.Length': 'SL', 'Sepal.Width':'SW', 'Petal.Length':'PL', 'Petal.Width':'PW'}, \
                    inplace=True)

#normalize data
dataNorm = preprocessing.MinMaxScaler().fit_transform(iris_data[['SL', 'SW', 'PL', 'PW']])
#data_dist = pdist(dataNorm, 'euclidean') #to scipy linkage

#use TSNE to reduce dimension
iris_data['Species'] = pd.Categorical(iris_data.Species)
iris_data['code'] = iris_data.Species.cat.codes
tsne = TSNE()
transformed = tsne.fit_transform(iris_data[['SL', 'SW', 'PL', 'PW', 'code']])
plt.figure(figsize=(16,10))
sns.scatterplot(
    x=transformed[:,0],y=transformed[:,1],
    palette=sns.color_palette("hls", 10),
    legend="full",
    alpha=0.3
);
plt.show()

#SKlearn
lst = []
for k in range(1,8):
    kmeans = KMeans(n_clusters=k, random_state=1).fit(iris_data[['SL', 'PL']])
    lst.append(np.sqrt(kmeans.inertia_))

model = KMeans(n_clusters=3, random_state=1).fit(iris_data[['SL', 'PL']])

fig, [ax1, ax2] = plt.subplots(nrows=1, ncols=2)
ax1.scatter(x=iris_data['SL'], y=iris_data['PL'], c=model.labels_)
ax2.plot(range(1, 8), lst, marker='s')
ax2.set_xlabel('$k$')
ax2.set_ylabel('$J(C_k)$')
plt.subplots_adjust(wspace=0.4)
plt.show()

#Scipy
model_scipy = linkage(iris_data[['SL', 'PL']], 'ward')
dg = dendrogram(model_scipy)
plt.show()
labels = fcluster(model_scipy, 9.5, criterion='distance')
sns.scatterplot(x=iris_data['SL'], y=iris_data['PL'], c=labels)
plt.show()
