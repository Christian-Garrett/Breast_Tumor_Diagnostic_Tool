import numpy as np

from Classification.code.Modular.ML_Pipeline.EDA import EDA
from Classification.code.Modular.ML_Pipeline.Prep import Prep
from Classification.code.Modular.ML_Pipeline.Gaussian import Gaussian
from Classification.code.Modular.ML_Pipeline.Hierarchical import Hierarchical
from Classification.code.Modular.ML_Pipeline.DBscan import DBscan
from Classification.code.Modular.ML_Pipeline.K_means import K_means



def main():

    ## Data Exploration - 

    drop_cols = ['ID']
    target_val = 'Diagnosis'
    path = 'Classification/code/Modular/Input/wdbc.csv'
    data = EDA(path, target_val, drop_cols)

    # Perform exploratory data analysis:
    data.explore()


    ## Data Preprocessing -

    raw_data_df, target_encoded_df = data.get_df_data()
    data_dict = {'raw':raw_data_df, 'tar':target_encoded_df}

    # list of features broken up by each category
    mean_list, se_list, worst_list = data.get_feature_category_lists()
    cat_dict = {'mean':mean_list, 'se':se_list, 'worst':worst_list}

    # skewed features that require distribution normalization
    skewed_features = ['Mean Compactness', 'Texture SE', 'Smoothness SE'
                      , 'Compactness SE', 'Concavity SE', 'Symmetry SE'
                      , 'Fractal Dimension SE', 'Worst Compactness'
                      , 'Worst Concavity', 'Worst Fractal Dimension']

    # prepare the model data, perform analysis and feature selection:
    model_data = Prep(data_dict, cat_dict, skewed_features)

    # get standardized and normalized selected features
    X_nrm = model_data.get_selected_norm_features() 
    X_std = model_data.get_selected_std_features() 
    data_dict = {'nrm':X_nrm, 'std':X_std}

    # get the dimension reduced selected data
    pca_dict, tsne_dict, umap_dict = model_data.get_reduced_selected_data()
    reduced_data_dict = {'pca':pca_dict, 'tsne':tsne_dict, 'umap':umap_dict}

    # get the data labels
    y = model_data.get_encoded_labels() 

    # the range of clusters to evaluate for k-means and the hierarchical models
    range_clusts = [2,5] 


    ## K-means Model - 

    # initialize and evaluate the k-means model. output elbow diagram and prediction plots:
    kmeans_mod = K_means(data_dict, reduced_data_dict, y, range_clusts)


    ## Hierarchical Model - 

    # hyperparameters:
    linkage_list = ['complete', 'average', 'ward', 'single']
    affinity_list = ['euclidean', 'cosine', 'manhattan', 'l1', 'l2']
    agglo_param_dict = {'linkage':linkage_list, 'affinity':affinity_list, 'clusters':range_clusts}

    # initialize, tune and evaluate the agglomerative model. output dendrograms and prediction plots:
    agglo_mod = Hierarchical(data_dict, reduced_data_dict, y, agglo_param_dict)


    ## DBScan Model - 

    # hyperparameters:
    eps_list = np.arange(.665, 1.0, .05)
    min_list = np.arange(5, 9, 1)
    metric_list = ['euclidean','l2'] # 'manhattan','cityblock','l1'

    dbscan_param_dict = {'eps':eps_list, 'min':min_list, 'met':metric_list}

    # initialize, tune and evaluate the dbscan model. output prediction plots:
    dbscan_mod = DBscan(data_dict, reduced_data_dict, y, dbscan_param_dict)


    ## Gaussian Model Evaluation - 

    # hyperparameters:
    cov_type_list = ['full', 'tied', 'diag', 'spherical']

    # initialize, tune and evaluate the gaussian model. output prediction plots:
    gauss_mod = Gaussian(data_dict, reduced_data_dict, y, cov_type_list)


   
main()

