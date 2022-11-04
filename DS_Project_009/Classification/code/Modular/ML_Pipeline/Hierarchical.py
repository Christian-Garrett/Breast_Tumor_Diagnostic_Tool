import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

from Classification.code.Modular.ML_Pipeline.Utils import *


class Hierarchical:


    def __init__(self, feat_dict, reduced_data_dict, labels, tuning_dict):
        self.X_nrm = feat_dict['nrm']
        self.X_std = feat_dict['std']
        self.y = labels
        self.tuning_dict = tuning_dict
        self.X_pca_2d = reduced_data_dict['pca']['pca_2d']
        self.X_pca_best = reduced_data_dict['pca']['pca_best']
        self.X_tsne_2d = reduced_data_dict['tsne']['tsne_2d']
        self.X_umap_2d = reduced_data_dict['umap']['umap_2d']
        self.X_umap_best = reduced_data_dict['umap']['umap_best']
        self.preds = self.generate_preds()
        self.best_param_list = self.generate_best_params_list()

        self.output_dendrogram('complete')
        self.output_dendrogram('average')
        self.output_dendrogram('ward')
        self.output_full_preds_plot()
        self.output_model_evaluation()



    # Hierarchical Hyperparameter Tuning
    def generate_best_params_list(self):

        linkage_list = self.tuning_dict['linkage']
        affinity_list = self.tuning_dict['affinity']
        clust_list = list(np.arange(self.tuning_dict['clusters'][0], self.tuning_dict['clusters'][1]))
     
        result_list = []
        for i in range(len(linkage_list)):
            for j in range(len(affinity_list)):
                for k in range(len(clust_list)):    
                    if ((linkage_list[i] == 'ward') and (affinity_list[j] != 'euclidean')):
                        continue
                    # Defining the agglomerative clustering and fit the model
                    agg_cluster = AgglomerativeClustering(linkage=linkage_list[i], affinity=affinity_list[j], n_clusters=clust_list[k])
                    clusters = agg_cluster.fit_predict(self.X_std)
                    ARI = metrics.adjusted_rand_score(self.y, clusters)
                    silhouette = metrics.silhouette_score(self.X_std, clusters, metric=affinity_list[j])
                    result_list.append([linkage_list[i], affinity_list[j], clust_list[k], set(clusters), ARI, silhouette])

        # sort the answers based on the ARI score
        for i in range(1, len(result_list)-1):
            ans = sorted(result_list,key=lambda x: x[4])

        print('\n')
        # remove duplicate ARI and silhouette scores
        seen = []
        newlist = []
        for i in range(1, len(ans)-1):
            unique_info = (ans[-i][4], ans[-i][5])
            if unique_info not in seen:
                newlist.append(ans[-i])
                seen.append(unique_info)
                print('linkage: {}, affinity: {}, num clusts: {}, set: {}, ARI: {}, Silhouette: {}'.format(ans[-i][0],ans[-i][1],ans[-i][2],ans[-i][3],ans[-i][4],ans[-i][5]))

        return newlist


    def output_model_evaluation(self):

        best_linkage = self.best_param_list[0][0]
        best_affinity = self.best_param_list[0][1]
        best_cluster = self.best_param_list[0][2]

        mod_array = []
        for nclusts in range(self.tuning_dict['clusters'][0], self.tuning_dict['clusters'][1]):
            agg_cluster = AgglomerativeClustering(linkage=best_linkage, affinity=best_affinity, n_clusters=nclusts)
            mod_array.append(agg_cluster)

        # get the number of dimensions for the best fit above the PCA threshold
        best_dim_num = len(self.X_pca_best[1])

        # print k-means final model evaluation with n-dimensional PCA reduction
        print_final_model_evaluation(mod_array, self.tuning_dict['clusters'], best_dim_num, self.X_pca_best, self.y, "Agglomerative")


    def output_dendrogram(self, type):

        output = "Classification/code/Modular/Output/Model_Analysis/models/hierachical/dendrograms/_{type}_method.png"
        plt.figure(figsize=(20,10))
        dendrogram(linkage(self.X_std, method=type))
        plt.savefig(output)
        plt.clf()


    def generate_preds(self):

        # Tuned agglomerative model with 'best features' for 2 cluster predictions   
        agg_cluster = AgglomerativeClustering(linkage='average', affinity='cosine', n_clusters=NUM_LABELS)
        preds = agg_cluster.fit_predict(self.X_std)

        return preds


    def output_full_preds_plot(self):

        pred_df = pd.DataFrame(data=self.preds)
        pred_df.columns = ['clust2']

        plot_dim_preds(pred_df, self.X_pca_2d, self.y, 'pca', "hierachical", 0)
        plot_dim_preds(pred_df, self.X_tsne_2d, self.y, 'tsne', "hierachical", 0)
        plot_dim_preds(pred_df, self.X_umap_2d, self.y, 'umap', "hierachical", 0)
