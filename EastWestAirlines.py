import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster	import	KMeans
from scipy.spatial.distance import cdist 


air=pd.read_csv("C:\\Users\\ADMIN\\Desktop\\Data_Science_Assig\\Clustering\\EastWestAirlines.csv")

def norm_func(i):
    x=(i-i.mean())/(i.std())
    return(x)
    
df_norm = norm_func(air.iloc[:,1:])

from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch
type(df_norm)
z = linkage(df_norm, method="complete",metric="euclidean")
plt.figure(figsize=(30,20));
plt.title('Hierarchical Clustering Dendrogram');
plt.xlabel('airdata');plt.ylabel('numbers')
sch.dendrogram(
    z,
    leaf_rotation=0.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()

from	sklearn.cluster	import	AgglomerativeClustering 
aaa = AgglomerativeClustering(n_clusters=3,	linkage='complete',affinity = "euclidean").fit(df_norm) 
cluster_labels=pd.Series(aaa.labels_)
air['clust']=cluster_labels
air = air.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11]]
air.head()

air['clust']=cluster_labels
air.iloc[:,2:].groupby(air.clust).median()

air.to_csv("airdata.csv",encoding="utf-8")


######################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster	import	KMeans
from scipy.spatial.distance import cdist 

air=pd.read_csv("C:\\Users\\ADMIN\\Desktop\\Data_Science_Assig\\Clustering\\EastWestAirlines.csv")

def norm_func(i):
    x=(i-i.mean())/(i.std())
    return(x)
    
df_norm = norm_func(air.iloc[:,1:])
from	sklearn.cluster	import	KMeans
from scipy.spatial.distance import cdist


# Generating random uniform numbers 
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


# Kmeans on University Data set 
#read file
#normalize the data

#df_norm = norm_func(air.iloc[:,1:])

df_norm.head(500)  # Top 10 rows

###### screw plot or elbow curve ############
k = list(range(200,1500))
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

model.labels_ # getting the labels of clusters assigned to each row 
md=pd.Series(model.labels_)  # converting numpy array into pandas series object 
air['clust']=md # creating a  new column and assigning it to new column 
df_norm.head()

air = air.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11]]

air.iloc[:,1:7].groupby(air.clust).mean()

air.to_csv("EastWestAirlines.csv")
