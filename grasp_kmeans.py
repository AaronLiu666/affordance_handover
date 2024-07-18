import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score




pc_path = '/home/mlx/GraspTTA/models/HO3D_Object_models/025_mug/resampled.npy'
grasp_path='/home/mlx/contact_graspnet/generated_grasps/111.npz'


points = np.load(pc_path)

# load .npz  and see whats inside
file = np.load(grasp_path)
grasps_list = file['arr_0']

data = grasps_list.reshape((200, 16))
data = data[:, 0 : 12]


# 确定最佳聚类数量
silhouette_scores = []
k_range = range(2, 11)  # 通常从2到10或更多

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data)
    score = silhouette_score(data, kmeans.labels_)
    silhouette_scores.append(score)

# 找到轮廓系数最高的聚类数量
best_k = k_range[np.argmax(silhouette_scores)]

# 用最佳聚类数量进行 K-means 聚类
kmeans = KMeans(n_clusters=best_k, random_state=42)
kmeans.fit(data)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# 打印最佳聚类数量和聚类结果
print("Best number of clusters:", best_k)
print("Cluster centers:\n", centroids)
print("Labels:\n", labels)

# 可视化轮廓系数
plt.plot(k_range, silhouette_scores, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Scores for different numbers of clusters')
plt.show()

# 使用 PCA 将数据降至2维，以便于可视化
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(data)

# 可视化聚类结果
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='viridis', marker='o')
plt.scatter(pca.transform(centroids)[:, 0], pca.transform(centroids)[:, 1], c='red', marker='x')
plt.title(f"K-means Clustering with PCA (Best k={best_k})")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()