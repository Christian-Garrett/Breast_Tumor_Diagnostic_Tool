import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

from Classification.code.Modular.ML_Pipeline.Utils import *





class K_means:


    def __init__(self, feat_dict, reduced_data_dict, labels, clust_range):
        self.X_nrm = feat_dict['nrm']
        self.X_std = feat_dict['std']
        self.cluster_range = clust_range
        self.y = labels
        self.X_pca_2d = reduced_data_dict['pca']['pca_2d']
        self.X_pca_best = reduced_data_dict['pca']['pca_best']
        self.ypred_pca = self.split_dim_predictions('pca')
        self.X_tsne_2d = reduced_data_dict['tsne']['tsne_2d']
        self.ypred_tsne = self.split_dim_predictions('tsne')
        self.X_umap_2d = reduced_data_dict['umap']['umap_2d']
        self.X_umap_best = reduced_data_dict['umap']['umap_best']
        self.ypred_umap = self.split_dim_predictions('umap')
        self.preds = self.generate_preds()

        self.output_elbow_diagram()
        self.output_split_preds_plot('pca')
        self.output_split_preds_plot('tsne')
        self.output_split_preds_plot('umap')
        self.output_full_preds_plot()
        self.output_model_evaluation()



    def output_model_evaluation(self, rand_state=123):

        mod_array = []
        for nclusts in range(self.cluster_range[0], self.cluster_range[1]):
            kmeans = KMeans(n_clusters=nclusts, random_state=rand_state)
            mod_array.append(kmeans)

        # get the number of dimensions for the best fit above the PCA threshold
        best_dim_num = len(self.X_pca_best[1])

        # print k-means final model evaluation with n-dimensional PCA reduction
        print_final_model_evaluation(mod_array, self.cluster_range, best_dim_num, self.X_pca_best, self.y, "K-means")


    def generate_preds(self, rand_state=123):

        # Kmeans model 'best features' 2 cluster predictions   
        kmeans = KMeans(n_clusters=NUM_LABELS, random_state=rand_state)
        preds = kmeans.fit_predict(self.X_std)

        return preds


    def output_full_preds_plot(self):

        pred_df = pd.DataFrame(data=self.preds)
        pred_df.columns = ['clust2']

        plot_dim_preds(pred_df, self.X_pca_2d, self.y, 'pca', "k_means", 0)
        plot_dim_preds(pred_df, self.X_tsne_2d, self.y, 'tsne', "k_means", 0)
        plot_dim_preds(pred_df, self.X_umap_2d, self.y, 'umap', "k_means", 0)


    def output_split_preds_plot(self, type):

        if type == 'pca':
            preds_df = self.ypred_pca
        elif type == 'tsne':
            preds_df = self.ypred_tsne
        elif type == 'umap':
            preds_df = self.ypred_umap

        # For each set of clusters, plot the clusters using the pca features for each sample.
        for cluster in range(self.cluster_range[0], self.cluster_range[1]):
    
            # Make a grid of subplots.
            plt.style.use("bmh")
            f, axarr = plt.subplots(1, 2)
    
            # Make a plot for each sample.
            for i in range(len(self.X_pca_2d[1])):
        
                # PCA-created features.
                x_sub = preds_df[f'{type}_f1_sample{i}']
                y_sub = preds_df[f'{type}_f2_sample{i}']
        
                # Cluster assignments.
                c = preds_df[f'clust{cluster}_sample{i}']
        
                # Assign the subplot to its place on the grid.
                axarr[i].scatter(x_sub, y_sub, c=c)
                axarr[i].set_title(f'sample {i}')

            
            # Space out the plots so that the headings don't overlap axis values.
            output = f'Classification/code/Modular/Output/Model_Analysis/dim_splits/{type}_{cluster}_clusters_split_visualization.png'
            plt.suptitle(f'{cluster} Clusters', fontsize=20)
            plt.tight_layout()
            plt.savefig(output)
            plt.clf()


    # Split the standardized and dimension reduced data in half
    def split_dim_predictions(self, prefix, test_size=0.5, rand_state=13579):

        # Initialize the proper reduced data
        if prefix == 'pca':
            X_data_dim = self.X_pca_2d
        elif prefix == 'tsne':
            X_data_dim = self.X_tsne_2d
        elif prefix == 'umap':
            X_data_dim = self.X_umap_2d

        # Remove the last row in the dataset for an even split
        X_data_std = self.X_std[:-1]
        X_data_dim = X_data_dim[:-1]


        X_half1, X_half2, X_pcahalf1, X_pcahalf2 = train_test_split(X_data_std
                                                                    , X_data_dim
                                                                    , test_size=test_size
                                                                    , random_state=rand_state)

        # Dataframe to store features and predicted cluster memberships
        ypred = pd.DataFrame()

        # Pass a list of tuples and a counter that increments each time we go through the loop. 
        # The tuples are the data to be used by k-means, and the PCA-derived features for graphing. 
        # We use k-means to fit a model to the data, then store the predicted values and the 
        # two-feature PCA solution in the data frame.
        for counter, data in enumerate([(X_half1, X_pcahalf1), (X_half2, X_pcahalf2)]):
    
            # Put the features into ypred
            ypred[f'{prefix}_f1' + '_sample' + str(counter)] = data[1][:, 0]
            ypred[f'{prefix}_f2' + '_sample' + str(counter)] = data[1][:, 1]
    
            # Generate cluster predictions and store them for 2-4 clusters
            for nclust in range(self.cluster_range[0], self.cluster_range[1]):
                pred = KMeans(n_clusters=nclust, random_state=123).fit_predict(data[counter])
                ypred['clust' + str(nclust) + '_sample' + str(counter)] = pred

        return ypred


    def output_elbow_diagram(self):

        # Create a K-means 'elbow diagram' to display SSE (inertia) values for a range of clusters:
        sse = []
        output = 'Classification/code/Modular/Output/Model_Analysis/models/k_means/Kmeans_Elbow_Diagram.png'
        for k in range(1, 11):
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(self.X_std)
            sse.append(kmeans.inertia_)

        plt.style.use("fivethirtyeight")
        plt.plot(range(1, 11), sse)
        plt.xticks(range(1, 11))
        plt.xlabel("Number of Clusters")
        plt.ylabel("SSE")
        plt.savefig(output)
        plt.clf()





