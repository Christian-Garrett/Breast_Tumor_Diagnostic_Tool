import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture

from Classification.code.Modular.ML_Pipeline.Utils import *



class Gaussian:


    def __init__(self, feat_dict, reduced_data_dict, labels, tuning_list):
        self.X_nrm = feat_dict['nrm']
        self.X_std = feat_dict['std']
        self.y = labels
        self.tuning_list = tuning_list
        self.X_pca_2d = reduced_data_dict['pca']['pca_2d']
        self.X_pca_best = reduced_data_dict['pca']['pca_best']
        self.X_tsne_2d = reduced_data_dict['tsne']['tsne_2d']
        self.X_umap_2d = reduced_data_dict['umap']['umap_2d']
        self.X_umap_best = reduced_data_dict['umap']['umap_best']
        self.best_cov_param = self.generate_best_cov_param()
        self.preds = self.generate_preds()

        self.output_full_preds_plot()
        self.output_model_evaluation()


    def generate_preds(self, rand_state=123):

       # Define and fit the GMM clustering final model:
        gmm_cluster = GaussianMixture(n_components=NUM_LABELS, random_state=rand_state
                                      , covariance_type=self.best_cov_param)
        preds = gmm_cluster.fit_predict(self.X_pca_best)

        return preds


    def output_full_preds_plot(self):

        pred_df = pd.DataFrame(data=self.preds)
        pred_df.columns = ['clust2']

        plot_dim_preds(pred_df, self.X_pca_2d, self.y, 'pca', "gaussian", 0)
        plot_dim_preds(pred_df, self.X_tsne_2d, self.y, 'tsne', "gaussian", 0)
        plot_dim_preds(pred_df, self.X_umap_2d, self.y, 'umap', "gaussian", 0)


    def output_model_evaluation(self):

        # Create a dictionary to store label predictions for 2D visualizations
        label_preds_dict = create_color_dict(NUM_LABELS, 0)

        for i in range(len(self.preds)):
            # increment the label dictionary prediction class counter
            label_preds_dict[COLOURS[int(self.y[i])]][self.preds[i]] += 1

        conf_mat = create_conf_mat(label_preds_dict)
        print(f'\nGaussian Final Model Confusion Matrix: \n{conf_mat}\n')

        # create a classification report that corresponds to the custom label ordering and cluster size created by the clustering algorithm  
        accuracy, score_matrix_df = create_score_matrix(conf_mat)
        print(f'Gaussian Final Model Score Matrix = \n{score_matrix_df}')
        print(f'Gaussian Final Model Accuracy Score = {accuracy}\n')

    # hyperparameter tuning the model
    def generate_best_cov_param(self):

        res_dict = {}
        for i in range(len(self.tuning_list)):
            # Defining and fit the Gaussian model for hyperparameter tuning
            gmm_cluster = GaussianMixture(n_components=NUM_LABELS, random_state=123, covariance_type=self.tuning_list[i])
            gaussian_clusters = gmm_cluster.fit_predict(self.X_std)
            ARI = metrics.adjusted_rand_score(self.y, gaussian_clusters)
            SIL = metrics.silhouette_score(self.X_std, gaussian_clusters, metric='euclidean')
            res_dict[self.tuning_list[i]] = {'ARI':ARI, 'SIL':SIL}
            print("Gaussian Model Tuning - Adjusted Rand Index with cov_type {}: {}".format(self.tuning_list[i], ARI))
            print("Gaussian Model Tuning - Silhoutte Score with cov_type {}: {}\n".format(self.tuning_list[i], SIL))

        # identify the covariance type parameter that yields the highest ARI score
        ARI_score_dict = {x:res_dict[x]['ARI'] for x in res_dict.keys()}
        result = max(ARI_score_dict, key=ARI_score_dict.get)

        return result




