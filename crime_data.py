import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv("C:\\Users\\ADMIN\\Desktop\\Data_Science_Assig\\Clustering\\crime_data.csv")

data.columns
data.isna().sum()
data.isnull().sum()

# normalization function 
def norm_func(i):
   x=(i-i.mean())/(i.std())
   return (x)

# normalized Data Frame only for numerical data
   df_norm = norm_func(data.iloc[:,1:])
    df_norm
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch
data.shape
type(df_norm)
help(linkage)
z=linkage(df_norm, method="complete",metric="euclidean")
plt.figure(figsize=(12,7));
plt.title('hierarchical Clustering Dendrogram');
plt.xlabel('city'); 
plt.ylabel('crime')

dend = sch.dendrogram(sch.linkage(df_norm, method='complete'))
sch.dendrogram(
    z,
    leaf_rotation=0.,  # rotates the x axis labels
    leaf_font_size=10.,  # font size for the x axis labels
)
plt.show()
from	sklearn.cluster	import	AgglomerativeClustering 
aaa	=	AgglomerativeClustering(n_clusters=3,	linkage='complete',affinity = "euclidean").fit(df_norm) 
cluster_labels=pd.Series(aaa.labels_)
aaa
aaa.labels_
cluster_labels
data['clust']=cluster_labels
data = data.iloc[:,[0,1,2,3,4]]
data.head()

data['clust']=cluster_labels
data.iloc[:,2:].groupby(data.clust).median()
data.to_csv("crime.csv",encoding="utf-8")

#############################

from	sklearn.cluster	import	KMeans
from scipy.spatial.distance import cdist

X = np.random.uniform(0,1,1000)
Y = np.random.uniform(0,1,1000)
df_xy =pd.DataFrame(columns=["X","Y"])
df_xy.X = X
df_xy.Y = Y
df_xy.plot(x="X",y = "Y",kind="scatter")
model1 = KMeans(n_clusters=5).fit(df_xy)
model1.labels_
model1.cluster_centers_
df_xy.plot(x="X",y = "Y",c=model1.labels_,kind="scatter",s=10,cmap=plt.cm.coolwarm)

def norm_func(i):
    x = (i-i.min())	/	(i.max()	-	i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(data.iloc[:,1:])
df_norm.head(25)

k = list(range(10,5))
k
TWSS = [] # variable for storing total within sum of squares for each kmeans 
for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    WSS = [] # variable for storing within sum of squares for each cluster 
    for j in range(i):
        WSS.append(sum(cdist(df_norm.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,df_norm.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))


# Scree plot 
plt.plot(k,TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS");plt.xticks(k)

# Selecting 5 clusters from the above scree plot which is the optimum number of clusters 
model=KMeans(n_clusters=5) 
model.fit(df_norm)

model.labels_  
md=pd.Series(model.labels_)   
data['clust']=md 
data.head()

data1 = data.iloc[:,[0,1,2,3,4,5]]

data.iloc[:,1:5].groupby(data.clust).mean()

data.to_csv("crime.csv")
