import random
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 

# Calculate distance
def calculate_distance(data_set, centroids, k):
	cla_list = []
	for data in data_set:
		diff = np.tile(data,(k,1)) - centroids # Copy 1x along the x-axis, which means no replication. Then replicate the result 2x along the y-axis.
		squared_diff = diff ** 2
		squared_dist = np.sum(squared_diff, axis=1)
		distance = squared_dist ** 0.5
		cla_list.append(distance)
	cla_list = np.array(cla_list)
	# print('cla_list: %s' % cla_list)
	return cla_list


# Calculate centroids
def classify(data_set, centroids, k):
	# Calculate the distance from data set to centroids
	cla_list = calculate_distance(data_set, centroids, k)
	# Grouping and calculate the new centroids
	min_distance_indices = np.argmin(cla_list, axis=1)
	new_centroids = pd.DataFrame(data_set).groupby(min_distance_indices).mean()
	new_centroids = new_centroids.values 
	# Calculate the changed value
	changed = new_centroids - centroids 
	return changed, new_centroids 


def kmeans(data_set, k): 
	# Randomly generate centroids
	centroids = random.sample(data_set, k) 
	# Update centroids until the changed value to 0
	changed,new_centroids = classify(data_set, centroids, k) 
	while np.any(changed != 0): 
		changed, new_centroids = classify(data_set, new_centroids, k)
	centroids = sorted(new_centroids.tolist())
	# Calculate each cluster regards to centroids
	cluster = []
	cla_list = calculate_distance(data_set, centroids, k)
	min_distance_indices = np.argmin(cla_list, axis=1) 
	for i in range(k): 
		cluster.append([]) 
	for i,j in enumerate(min_distance_indices):
		cluster[j].append(data_set[i]) 
	return centroids, cluster


if __name__=='__main__':
	data_set = [[1, 1], [1, 3], [2, 1], [5, 4], [6, 3], [6, 5]]
	centroids, cluster = kmeans(data_set, 2)
	print('Centroids: %s' % centroids)
	print('Clusters %s' % cluster)
	for i in range(len(data_set)):
		plt.scatter(data_set[i][0], data_set[i][1], marker='o', color ='green', s=40, label='Original point')
	for j in range(len(centroids)):
		plt.scatter(centroids[j][0], centroids[j][1], marker='x', color='red', s=50, label='Centroid')
	plt.show()
