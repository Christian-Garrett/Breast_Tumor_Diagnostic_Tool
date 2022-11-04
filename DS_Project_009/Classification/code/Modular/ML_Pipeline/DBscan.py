import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN

from Classification.code.Modular.ML_Pipeline.Utils import *



class DBscan:


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
        self.best_param_list = self.generate_best_params_list(ari_thresh=.3, max_iters=7)

        if len(self.best_param_list) > 0:
            self.dbscan_mod = self.output_full_preds_plots()
            self.output_model_evaluation()
        else:
            print('\nThe DBScan Model did not resolve succesfully.\n')



    def output_model_evaluation(self):

        labels = self.dbscan_mod.labels_
        label_pred_offset = (1 if -1 in labels else 0)
        label_count = len(set(labels))
        n_clusters_ = label_count - label_pred_offset
        n_noise_ = list(labels).count(-1)


        params = self.dbscan_mod.get_params()
        eps = params['eps']
        samp = params['min_samples']
        metric = params['metric']

        print(f'\nResults for DBscan Model with Parameters - eps: {eps}, min: {samp}, metric: {metric}')
        print("Estimated number of clusters: %d" % n_clusters_)
        print("Estimated number of noise points: %d" % n_noise_)
        print("Homogeneity: %0.3f" % metrics.homogeneity_score(self.y, labels))
        print("Completeness: %0.3f" % metrics.completeness_score(self.y, labels))
        print("V-measure: %0.3f" % metrics.v_measure_score(self.y, labels))
        print("Adjusted Mutual Information: %0.3f" % metrics.adjusted_mutual_info_score(self.y, labels))
        print("\nAdjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(self.y, labels))
        print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(self.X_umap_best, labels, metric='euclidean'))


        # Create a dictionary to store label predictions for 2D visualizations
        label_preds_dict = create_color_dict(label_count, label_pred_offset)

        for i in range(len(labels)):
            # increment the label dictionary prediction class counter
            label_preds_dict[COLOURS[int(self.y[i])]][labels[i]] += 1


        conf_mat = create_conf_mat(label_preds_dict)
        print(f'\nDBScan Final Model Confusion Matrix: \n{conf_mat}\n')

        # create a classification report that corresponds to the custom label ordering and cluster size created by the clustering algorithm  
        accuracy, score_matrix_df = create_score_matrix(conf_mat)
        print(f'DBScan Final Model Score Matrix = \n{score_matrix_df}')
        print(f'DBScan Final Model Accuracy Score = {accuracy}\n')




    def plot_dbscan_preds(self, db_clust):

        core_samples_mask = np.zeros_like(db_clust.labels_, dtype=bool)
        core_samples_mask[db_clust.core_sample_indices_] = True
        labels = db_clust.labels_

        # Number of clusters in labels, ignoring noise if present.
        label_pred_offset = (1 if -1 in labels else 0)
        unique_labels = set(labels)
        label_count = len(unique_labels)
        n_clusters_ = label_count - label_pred_offset
        # n_noise_ = list(labels).count(-1)

        # Black removed and is used for noise instead.
        colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]

            class_member_mask = labels == k

            xy = self.X_umap_best[class_member_mask & core_samples_mask]
            plt.plot(
                xy[:, 0],
                xy[:, 1],
                "o",
                markerfacecolor=tuple(col),
                markeredgecolor="k",
                markersize=14,
            )

            xy = self.X_umap_best[class_member_mask & ~core_samples_mask]
            plt.plot(
                xy[:, 0],
                xy[:, 1],
                "o",
                markerfacecolor=tuple(col),
                markeredgecolor="k",
                markersize=6,
            )

        output='Classification/code/Modular/Output/Model_Analysis/models/db_scan/dbscan_umap_best_full_visualization.png'
        plt.title("Final DBScan Model - Estimated number of clusters: %d" % n_clusters_)
        plt.savefig(output)
        plt.clf()


    def output_full_preds_plots(self):

        # dbscan_cluster = DBSCAN(eps=.965, min_samples=7.5, metric='l2')

        # Plot DBscan Dimensionality Reduction Visualization for 'N' Clusters:   
        eps_val = self.best_param_list[0]['EPS']
        min_val = self.best_param_list[0]['MIN']
        met_val = self.best_param_list[0]['MET']
        dbscan_cluster = DBSCAN(eps=eps_val, min_samples=min_val, metric=met_val)
        
        labels = dbscan_cluster.fit_predict(self.X_umap_best)

        # labels = dbscan_cluster.labels_
        label_pred_offset = (1 if -1 in labels else 0)

        pred_df = pd.DataFrame(data=labels)
        pred_df.columns = ['clust2']

        plot_dim_preds(pred_df, self.X_pca_2d, self.y, 'pca', "db_scan", label_pred_offset)
        plot_dim_preds(pred_df, self.X_tsne_2d, self.y, 'tsne', "db_scan", label_pred_offset)
        plot_dim_preds(pred_df, self.X_umap_2d, self.y, 'umap', "db_scan", label_pred_offset)

        self.plot_dbscan_preds(dbscan_cluster)

        return dbscan_cluster


    def generate_best_params_list(self, clust_thresh=5, ari_thresh=.2, sil_thresh=.1, max_iters=5):

        eps_dist = self.tuning_dict['eps']
        min_samp = self.tuning_dict['min']
        met_list = self.tuning_dict['met']

        res_list = []
        for k in range(max_iters):
            result_array = []
            for eps in eps_dist:
                for samp in min_samp:
                    for metric in met_list:
                        db = DBSCAN(eps=eps, min_samples=samp, metric=metric).fit(self.X_umap_best)
                        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
                        core_samples_mask[db.core_sample_indices_] = True
                        labels = db.labels_

                        unique_labels = len(set(labels))
                        if (unique_labels < 2):
                            continue

                        # Number of clusters in labels, ignoring noise if present.
                        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
                        n_noise_ = list(labels).count(-1)

                        ari_index = metrics.adjusted_mutual_info_score(self.y, labels)
                        sil_score = metrics.silhouette_score(self.X_umap_best, labels)

                        if n_clusters_ < clust_thresh and n_clusters_ > 2 and ari_index > ari_thresh: # and sil_score > sil_thresh:
                            entry_dict = {}
                            entry_dict['SIL'] = sil_score
                            entry_dict['ARI'] = ari_index
                            entry_dict['EPS'] = eps
                            entry_dict['MIN'] = samp
                            entry_dict['MET'] = metric
                            entry_dict['NUM'] = n_clusters_

                            result_array.append(entry_dict)

            if not len(result_array) == 0:
                res_list = sorted(result_array, key = lambda i: i['ARI'], reverse=True)
                break

        return res_list


