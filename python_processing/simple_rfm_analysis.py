import sys
import pandas as pd
import math
import numpy as np
import datetime as dt
import random
import json
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

retail_df = pd.read_csv(str(sys.argv[1]))

# below reference date will be inputted by the user
rcv_obj = json.loads(sys.argv[2])

ref_year = int(str(rcv_obj['year']))
ref_month = int(str(rcv_obj['month']))
ref_day = int(str(rcv_obj['day']))

retail_df['InvoiceDate'] = pd.to_datetime(retail_df['InvoiceDate'])

retail_df = retail_df[retail_df['Quantity'] > 0]
retail_df = retail_df[retail_df['UnitPrice'] > 0]
retail_df = retail_df[retail_df['CustomerKey'].notnull()]

retail_df['sale_amount'] = retail_df['Quantity'] * retail_df['UnitPrice']
retail_df['CustomerKey'] = retail_df['CustomerKey'].astype(int)

aggregations = {
    'InvoiceDate': 'max',
    'InvoiceNo': 'count',
    'sale_amount': 'sum'
}
cust_df = retail_df.groupby('CustomerKey').agg(aggregations)
cust_df = cust_df.rename(columns = {'InvoiceDate': 'Recency',
                                   'InvoiceNo': 'Frequency',
                                   'sale_amount': 'Monetary'})
cust_df = cust_df.reset_index()

cust_df['Recency'] = dt.datetime(ref_year, ref_month, ref_day) - cust_df['Recency']
cust_df['Recency'] = cust_df['Recency'].apply(lambda x: x.days + 1)

cust_df['Recency_log'] = np.log1p(cust_df['Recency'])
cust_df['Frequency_log'] = np.log1p(cust_df['Frequency'])
cust_df['Monetary_log'] = np.log1p(cust_df['Monetary'])

X_features = cust_df[['Recency_log', 'Frequency_log', 'Monetary_log']].values
ss = StandardScaler().fit(X_features)
X_features_scaled = ss.transform(X_features)

n_clusters_list = range(2, 6)
silhouette_score_list = []

# find n_clusters that maximized silhouette score
for i in n_clusters_list:
    kmeans = KMeans(n_clusters=i)
    labels = kmeans.fit_predict(X_features_scaled)
    silhouette_score_list.append(silhouette_score(X_features_scaled, labels))

# show below on the website
optimal_sil_score = round(max(silhouette_score_list), 2)
optimal_n_clusters = n_clusters_list[np.argmax(silhouette_score_list)]

kmeans = KMeans(n_clusters=optimal_n_clusters)
labels = kmeans.fit_predict(X_features_scaled)
cust_df['ClusterLabel'] = labels

label_list = np.unique(cust_df['ClusterLabel'])

# below generates csv file with x, y, z data 
# one file per cluster
file_names = []
for label in label_list:
    label_cluster = cust_df[cust_df['ClusterLabel']==label]
    rand_num = random.randint(1,10000)
    time_num = time.time()
    file_stuff = 'Data2Plot_' + str(rand_num) + '_' + str(time_num) + '.csv'
    file_name = './assets/output/' + file_stuff
    label_cluster[['Recency_log', 'Frequency_log', 'Monetary_log']].to_csv(file_name)
    file_names.append(file_stuff)

'''
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

markers = ['o', '^', '1', 's', 'p', 'P', '*', '+', 'd']

for label in label_list:
    label_cluster = cust_df[cust_df['ClusterLabel']==label]
    ax.scatter(xs=label_cluster['Recency_log'], ys=label_cluster['Frequency_log'], zs=label_cluster['Monetary_log'], edgecolor='k', marker=markers[label])

ax.set_xlabel('Recency_log')
ax.set_ylabel('Frenquency_log')
ax.set_zlabel('MonetaryValue_log')

plt.savefig('3d_plot')
'''

# this output_df should be returned as analysis result
output_df = cust_df[['CustomerKey', 'ClusterLabel']]
output_df_name = './assets/output/ClusteringResult_' + str(rand_num) + '_' + str(time_num) + '.csv'
output_df.to_csv(output_df_name, index=False)

labelling_info = np.unique(kmeans.labels_, return_counts=True)

# inverse log transformation
cluster_centers = np.expm1(kmeans.cluster_centers_)
# inverse normalization
cluster_centers = ss.inverse_transform(cluster_centers)


num_point_cluster = {}
centroids = {}
for i in range(optimal_n_clusters):
  num_point_cluster[i] = labelling_info[1][i]
  centroids[i] = {'Recency': cluster_centers[i, 0],
                  'Frequency': cluster_centers[i, 1],
                  'Monetary': cluster_centers[i, 2]}

# index_list = []
# for i in range(optimal_n_clusters):
#     index_list.append('Cluster ' + str(i))
# output_df2 = pd.DataFrame(labelling_info[1].tolist(), columns={'Number of Customers'}, index=index_list)


# output_df3 = pd.DataFrame(cluster_centers, columns = ['Recency', 'Frequency', 'Monetary'], index=index_list)

# str1 = '"highest_silhouette_score": ' + "'" + str(optimal_sil_score) + "'"
# str2 = '"optimal_n_clusters": ' + "'" + str(optimal_n_clusters) + "'"
# str3 = '"clustering_result_csv": ' + "'" + str(output_df_name) + "'"
# str4 = '"plot_data_csv_names": ' + "'" + str(file_names) + "'"
# str5 = '"num_of_data_points_in_each_cluster": ' + "'" + str(num_point_cluster) + "'"
# str6 = '"centroid_info": ' + "'" + str(centroids) + "'"

# obj = '{' + str1 + ',' + str2 + ',' + str3 + ',' + str4 + ',' + str5 + ',' + str6 + '}'

str1 = str(optimal_sil_score)
str2 = str(optimal_n_clusters) 
str3 = str(output_df_name)
str4 = str(file_names)
str5 = str(num_point_cluster)
str6 = str(centroids)

obj = str1 + '||' + str2 + '||' + str3 + '||' + str4 + '||' + str5 + '||' + str6

# teststr = '{"user_id": 2131, "name": "John", "gender": 0,  "thumb_url": "sd", "money": 23, "cash": 2, "material": 5}'

# arr_ret = [optimal_sil_score, optimal_n_clusters, output_df_name, file_names, num_point_cluster, centroids]

print(obj)

sys.stdout.flush()
