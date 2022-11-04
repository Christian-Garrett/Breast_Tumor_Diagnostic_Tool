import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import umap
import seaborn as sns

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.cluster import KMeans

from Classification.code.Modular.ML_Pipeline.Utils import *



def get_subplot_coords(curr_num, col_count):

    y = (curr_num // col_count)
    x = curr_num - (y * col_count)

    return y, x


def corrFilter(x: pd.DataFrame, bound: float):

    xCorr = x.corr()
    xFiltered = xCorr[((xCorr >= bound) | (xCorr <= -bound)) & (xCorr !=1.000)]
    xFlattened = xFiltered.unstack().sort_values().drop_duplicates()
    result = xFlattened.dropna()

    return result


def logit_pvalue(model, x):
    """
    Calculate z-scores for scikit-learn LogisticRegression.
    params
       (1) model:  fitted sklearn.linear_model.LogisticRegression with intercept and large C
       (2) x:      matrix on which the model was fit
    This function uses asymtptics for maximum likelihood estimates.
    """

    from scipy.stats import norm

    p = model.predict_proba(x)
    n = len(p)
    m = len(model.coef_[0]) + 1
    coefs = np.concatenate([model.intercept_, model.coef_[0]])
    x_full = np.matrix(np.insert(np.array(x), 0, 1, axis = 1))
    ans = np.zeros((m, m))
    for i in range(n):
        ans = ans + np.dot(np.transpose(x_full[i, :]), x_full[i, :]) * p[i,1] * p[i, 0]
    vcov = np.linalg.inv(np.matrix(ans))
    se = np.sqrt(np.diag(vcov))
    t =  coefs/se  
    p = (1 - norm.cdf(abs(t))) * 2
    return p


def generate_accuracy_and_heatmap(mod, x, y):
    
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.metrics import f1_score,confusion_matrix
    from sklearn.metrics import classification_report


    # cm = confusion_matrix(y,model.predict(x))
    # sns.heatmap(cm,annot=True,fmt="d")
    pred_dict = {}
    pred_x = mod.predict(x)
    ac = accuracy_score(y.values, pred_x)
    f_score = f1_score(y.values, pred_x)
    cross_tab = pd.crosstab(pd.Series(mod.predict(x)), y.values, colnames=['Predicted'], rownames=['Actual'])
    class_report = classification_report(y.values, pred_x)

    pred_dict.update(Accuracy=ac, F1=f_score, Conf=cross_tab, Report=class_report)

    return pred_dict


class Prep:


    def __init__(self, data, cats, skew_feats=None, out_lims=None, pca_thresh=.95):

        self.raw_data = data['raw']
        self.tar_enc_data = data['tar']
        self.tar_name = self.raw_data.columns[0]
        self.cat_dict = cats
        if out_lims:
            self.lim = out_lims
        else:
            self.lim = []
        if skew_feats:
            self.skewed_feats = skew_feats
        else:
            self.skewed_feats = []
        self.pca_threshold = pca_thresh * 100
        self.std_X, self.nrm_X = self.encode_data()
        self.enc_y = self.tar_enc_data[self.tar_name]

        self.multicollinearity_dict = self.create_multicollinearity_list() 
        self.variance_inflation_factor_list = self.create_VIF_list() 
        self.target_collinearity_list = self.create_target_collinearity_list() 
        self.log_reg_dict = self.create_log_reg_dict() 
        self.selectKbest_dict = self.create_k_best_dict()
        self.RFE_dict = self.create_RFE_dict()
        self.selected_features = self.RFE_dict['rfecv']['top_ranked'].values
        self.X_nrm_selected = self.nrm_X[self.selected_features]
        self.X_std_selected = self.std_X[self.selected_features]
        self.best_dim_num = self.output_culm_variance()
        self.pca_data_dict = self.create_reduced_data_dict('pca')
        self.tsne_data_dict = self.create_reduced_data_dict('tsne')
        self.umap_data_dict = self.create_reduced_data_dict('umap')
        self.centroids_std_df = self.output_centroid_analysis()


    def create_reduced_data_dict(self, type):

        res_dict = {}
        if type == 'pca':
            pca = PCA(n_components = 2)
            X_pca_2d = pca.fit_transform(self.X_std_selected)

            pca = PCA(n_components = self.best_dim_num)
            X_pca_best = pca.fit_transform(self.X_std_selected)

            res_dict = {'pca_2d': X_pca_2d, 'pca_best': X_pca_best}

        elif type == 'tsne':
            tsne = TSNE(n_components = 2, verbose=0, perplexity=40, n_iter=300)
            X_tsne_2d = tsne.fit_transform(self.X_nrm_selected)

            res_dict = {'tsne_2d': X_tsne_2d}

        elif type == 'umap':
            X_umap_2d = umap.UMAP(n_components = 2, n_neighbors=5, min_dist=0.3, metric='correlation').fit_transform(self.X_nrm_selected)
            X_umap_best = umap.UMAP(n_components = self.best_dim_num, n_neighbors=5, min_dist=0.3, metric='correlation').fit_transform(self.X_nrm_selected)

            res_dict = {'umap_2d': X_umap_2d, 'umap_best': X_umap_best}

        return res_dict


    def output_centroid_analysis(self):

        # Full Data K-means Centroid Analysis:
        kmeans = KMeans(n_clusters=NUM_LABELS).fit(self.std_X)
        memb =  pd.Series(kmeans.labels_, index=self.std_X.index)

        clust = ['Cluster {}'.format(i) for i in range(NUM_LABELS)]
        Centroids_std = pd.DataFrame(0.0, index=clust, columns=self.std_X.columns)

        for i in range(NUM_LABELS):
            BM = memb==i
            Centroids_std.iloc[i] = self.std_X[BM].mean(axis=0)
           
        output = 'Classification/code/Modular/Output/Model_Analysis/Kmeans_Centroid_Analysis.png'
        plt.figure(figsize=(30,5))
        sns.heatmap(Centroids_std, linewidths=.5, annot=True, cmap='Purples')
        plt.savefig(output)
        plt.clf()

        return Centroids_std


    def output_culm_variance(self):

        # create a covariance matrix out of the selected (best) features:
        covar_matrix = PCA(n_components = len(self.selected_features)) 

        #calculate variance ratios
        covar_matrix.fit(self.X_std_selected)
        variance = covar_matrix.explained_variance_ratio_ 

        #cumulative sum of variance explained with [n] features
        var=np.cumsum(np.round(covar_matrix.explained_variance_ratio_, decimals=3)*100)
        # print(f'\nCulmulative sum of variance explained: {var}') 
        res_val = list(filter(lambda i: i > self.pca_threshold, var))[0]
        num_dims = (list(var).index(res_val)) + 1
        print(f'\nThe final model will be reduced to {num_dims} dimensions in order to retain over {res_val} percent of the model information\n')

        output = 'Classification/code/Modular/Output/Model_Analysis/PCA_Analysis.png'
        plt.ylabel('% Variance Explained')
        plt.xlabel('# of Features')
        plt.title('PCA Analysis')
        plt.ylim(30,100.5)
        plt.style.context('seaborn-whitegrid')
        plt.plot(var)
        plt.savefig(output)
        plt.clf()

        return num_dims


    def create_log_reg_dict(self):

        clf_lr = LogisticRegression()      
        y = self.tar_enc_data[self.tar_name]


        # logistic regression accuracy with all features to establish baseline feature significance
        X = self.tar_enc_data.drop(columns=[self.tar_name])

        x_tra, x_tes, y_tra, y_tes = train_test_split(X, y, test_size=0.33, random_state=8)
        lr_baseline_model = clf_lr.fit(x_tra,y_tra)
        p_vals = logit_pvalue(lr_baseline_model, x_tra)

        base_preds = generate_accuracy_and_heatmap(lr_baseline_model, x_tes, y_tes)
        print('\nBaseline Logistic Confusion Matrix:\n{}\n' .format(base_preds['Conf']))
        print('Baseline Logistic Classification Report:\n{}\n' .format(base_preds['Report']))
        coef_df = pd.DataFrame(np.array([lr_baseline_model.feature_names_in_, lr_baseline_model.coef_[0], p_vals[1:]])).T
        coef_df.columns = ['Feature', 'Coefficients', 'p-Values']
        coef_df.sort_values(by=['p-Values'], inplace=True)
        print(f'Baseline Logistic Regression Coefficient p-Values: \n{coef_df}\n')

        All_Feats_Dict = {'model': lr_baseline_model
                          , 'p_vals': p_vals
                          , 'conf_mat': base_preds['Conf']
                          , 'class_rep': base_preds['Report']
                          , 'coefs': coef_df
                          }


        # logistic regression test after removing the best highly multicollinear features above 90%
        dropcols = list(self.multicollinearity_dict['Remove'])
        dropcols.insert(0, self.tar_name)
        X = self.tar_enc_data.drop(columns=dropcols, axis=1)

        x_tra, x_tes, y_tra, y_tes = train_test_split(X, y, test_size=0.33, random_state=8)
        lr_baseline_model = clf_lr.fit(x_tra,y_tra)
        p_vals = logit_pvalue(lr_baseline_model, x_tra)

        base_preds = generate_accuracy_and_heatmap(lr_baseline_model, x_tes, y_tes)
        print('\nBaseline Logistic Confusion Matrix:\n{}\n' .format(base_preds['Conf']))
        print('Filtered Baseline Logistic Classification Report:\n{}\n' .format(base_preds['Report']))
        coef_df = pd.DataFrame(np.array([lr_baseline_model.feature_names_in_, lr_baseline_model.coef_[0], p_vals[1:]])).T
        coef_df.columns = ['Feature', 'Coefficients', 'p-Values']
        coef_df.sort_values(by=['p-Values'], inplace=True)
        print(f'Filtered Baseline Logistic Regression Coefficient p-Values: \n{coef_df}\n')

        Trunc_Feats_Dict = {'model': lr_baseline_model
                          , 'p_vals': p_vals
                          , 'conf_mat': base_preds['Conf']
                          , 'class_rep': base_preds['Report']
                          , 'coefs': coef_df
                          }

        log_reg_scores_dict = {'All': All_Feats_Dict, 'Trunc': Trunc_Feats_Dict}

        return log_reg_scores_dict


    def create_k_best_dict(self, k_val=12):

        X = self.tar_enc_data.drop(columns=[self.tar_name])
        y = self.tar_enc_data[self.tar_name]

        x_tra, x_tes, y_tra, y_tes = train_test_split(X, y, test_size=0.33, random_state=8)
        select_feature = SelectKBest(chi2, k=k_val).fit(x_tra, y_tra)

        selected_features_df = pd.DataFrame({'Feature':list(x_tra.columns)
                                             ,'Scores':select_feature.scores_})
        selected_features_df.sort_values(by='Scores', ascending=False, inplace=True)
        print(f'\nSelect {k_val} Best Features: \n{selected_features_df}\n')

        x_train_chi = select_feature.transform(x_tra)
        x_test_chi = select_feature.transform(x_tes)

        clf_lr = LogisticRegression()      
        lr_chi_model = clf_lr.fit(x_train_chi, y_tra)
        p_vals = logit_pvalue(lr_chi_model, x_train_chi)

        preds = generate_accuracy_and_heatmap(lr_chi_model, x_test_chi, y_tes)
        print('\nSelect {} Best Confusion Matrix:\n{}\n' .format(k_val, preds['Conf']))
        print('Select {} Best Classification Report:\n{}\n' .format(k_val, preds['Report']))

        result_dict = {'model': lr_chi_model
                          , 'conf_mat': preds['Conf']
                          , 'class_rep': preds['Report']
                          , 'p-vals': p_vals
                          , 'sorted_feats': selected_features_df
                          }

        return result_dict


    def create_RFE_dict(self):

        x_tra, x_tes, y_tra, y_tes = train_test_split(self.std_X, self.enc_y, test_size=0.33, random_state=8)
        
        # recursive feature elimination
        clf_lr = LogisticRegression()      
        rfe = RFE(estimator=clf_lr, step=1)
        rfe = rfe.fit(x_tra, y_tra)

        selected_rfe_features = pd.DataFrame({'Feature':list(x_tra.columns)
                                              ,'Ranking':rfe.ranking_})
        selected_rfe_features.sort_values(by='Ranking', inplace=True)
        print(f'\nRFE Ranked Features: \n{selected_rfe_features}\n')
        rfe_top_ranked = list(selected_rfe_features[selected_rfe_features['Ranking'] == 1]['Feature'].values)
                                     
        x_train_rfe = rfe.transform(x_tra)
        x_test_rfe = rfe.transform(x_tes)

        lr_rfe_model = clf_lr.fit(pd.DataFrame(x_train_rfe, columns=rfe_top_ranked), y_tra)
        p_vals = logit_pvalue(lr_rfe_model, x_train_rfe)

        preds = generate_accuracy_and_heatmap(lr_rfe_model, x_test_rfe, y_tes)
        print('\nRFE Confusion Matrix:\n{}\n' .format(preds['Conf']))
        print('RFE Classification Report:\n{}\n' .format(preds['Report']))
        coef_df = pd.DataFrame(np.array([lr_rfe_model.feature_names_in_, lr_rfe_model.coef_[0], p_vals[1:]])).T
        coef_df.columns = ['Feature', 'Coefficients', 'p-Values']
        coef_df.sort_values(by=['p-Values'], inplace=True)
        print(f'RFE Logistic Regression Coefficient p-Values: \n{coef_df}\n')

        rfe_dict = {'model': lr_rfe_model
                          , 'conf_mat': preds['Conf']
                          , 'class_rep': preds['Report']
                          , 'p_vals': p_vals
                          , 'top_ranked': rfe_top_ranked
                          }

        # recursive feature elimination cross validation
        rfecv = RFECV(estimator=clf_lr, step=1, cv=5, scoring='accuracy')
        rfecv = rfecv.fit(x_tra, y_tra)

        print('\nOptimal number of features after RFECV :', rfecv.n_features_)
        rfecv_top_ranked = x_tra.columns[rfecv.support_]
        print('RFECV Best features :\n', rfecv_top_ranked)
        # print('Grid scores :', rfecv.cv_results_)

        output = 'Classification\code\Modular\Output\Model_Analysis\RFECV_grid_scores.png'
        plt.figure()
        plt.xlabel("Number of features selected")
        plt.ylabel("Cross validation score of number of selected features")
        plt.plot(range(1, len(rfecv.cv_results_['mean_test_score']) + 1), rfecv.cv_results_['mean_test_score'])
        plt.savefig(output)
        plt.clf()

        x_train_rfecv = rfecv.transform(x_tra)
        x_test_rfecv = rfecv.transform(x_tes)

        lr_rfecv_model = clf_lr.fit(pd.DataFrame(x_train_rfecv, columns=rfecv_top_ranked), y_tra)
        p_vals = logit_pvalue(lr_rfecv_model, x_train_rfecv)

        preds = generate_accuracy_and_heatmap(lr_rfecv_model, x_test_rfecv, y_tes)
        print('\nRFECV Confusion Matrix:\n{}\n' .format(preds['Conf']))
        print('RFECV Classification Report:\n{}\n' .format(preds['Report']))
        coef_df = pd.DataFrame(np.array([lr_rfecv_model.feature_names_in_, lr_rfecv_model.coef_[0], p_vals[1:]])).T
        coef_df.columns = ['Feature', 'Coefficients', 'p-Values']
        coef_df.sort_values(by=['p-Values'], inplace=True)
        print(f'RFECV Logistic Regression Coefficient p-Values: \n{coef_df}\n')

        rfecv_dict = {'model': lr_rfecv_model
                          , 'conf_mat': preds['Conf']
                          , 'class_rep': preds['Report']
                          , 'p_vals': p_vals
                          , 'top_ranked': rfecv_top_ranked
                          }

        result_dict = {'rfe': rfe_dict, 'rfecv': rfecv_dict}

        selected_feats = rfecv_dict['top_ranked'].values
        print(f'\nFinal model {len(selected_feats)} selected features: \n{selected_feats}\n')

        return result_dict


    def create_target_collinearity_list(self, thresh = 0.6):

        # get target collinearity information
        Diagnosis_Corr = self.tar_enc_data.corr()[self.tar_name].iloc[1:]
        cor_thresh = Diagnosis_Corr[(Diagnosis_Corr >= thresh)].sort_values()
        print(f'\nTarget Correlations For {len(cor_thresh)} Features Above {thresh}: \n{cor_thresh}')

        return cor_thresh


    def create_VIF_list(self):

        X = self.raw_data.drop(columns=[self.tar_name])

        # get the variance inflation factor data as a measure of multi-collinearity
        vif_info = pd.DataFrame()
        vif_info['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        vif_info['Column'] = X.columns
        VIF_list = vif_info.sort_values('VIF', ascending=False)
        print(VIF_list)

        return VIF_list


    def create_multicollinearity_list(self, thresh = 0.9):

        # get multicollineary information
        corr_result = corrFilter(self.raw_data, thresh)

        # flatten the multi row index and filter out the redundant feature names
        index_values = corr_result.index.values
        np_index_values = np.array(list(corr_result.index.values))
        rowcol1 = np_index_values[:,0]
        rowcol2 = np_index_values[:,1]
        np_all_values = np.concatenate([rowcol1, rowcol2])
        value_list = list(set(np_all_values))
        print(f'\n Multicollinearity For {len(value_list)} Features Above {thresh}: \n{corr_result}')

        result_dict = {'Full':corr_result, 'Flat':value_list, 'Remove':rowcol1}

        return result_dict


    def output_transformed_features(self, trans_dict):

        sub_rows = 2
        sub_cols = 5
        # output histogram of the transformed data by feature categories
        output = 'Classification/code/Modular/Output/Data_Exploration/Log_Transformations.png'
        plt.style.use("bmh")
        fig, axarr = plt.subplots(sub_rows, sub_cols)
        for feature in self.skewed_feats:
            row, col = get_subplot_coords(self.skewed_feats.index(feature), sub_cols)
            axarr[row][col].hist(trans_dict[feature])
            axarr[row][col].set_title(feature, fontsize=8)
            axarr[row][col].tick_params(labelrotation=45)
        fig.tight_layout()
        plt.savefig(output)
        plt.clf()


    def winsorize_data(self, X_data):

        # trim away outliers using winsorization
        X_data = pd.DataFrame(winsorize(X_data.to_numpy(), limits=[self.lim[0], self.lim[1]], inplace=True), columns = X_data.columns)

        return X_data


    def normalize_distributions(self, X_data):

        log_trans = {}
        power = PowerTransformer(method='yeo-johnson', standardize=True)

        for feature in self.skewed_feats:

            dat_res = np.array(X_data[feature])
            dat_res = dat_res.reshape((len(dat_res),1))
            fitted_data = power.fit_transform(dat_res)
            log_trans[feature]=pd.DataFrame(fitted_data)

        for feature in log_trans.keys():
            X_data[feature] = log_trans[feature]

        self.output_transformed_features(log_trans)

        return X_data


    def encode_data(self):

        data_df = self.tar_enc_data
        X = data_df.drop(columns=[self.tar_name])

        # normalize skewed distributions if necessary
        if(self.skewed_feats):
            X = self.normalize_distributions(X)

        # winsorize the data if necessary
        if(self.lim):
            X = self.winsorize_data(X)

        # standardize the feature data
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X)
        X_std = pd.DataFrame(np.array(X_std), columns=X.columns)

        # normalize the feature data
        X_nrm = normalize(X)
        X_nrm = pd.DataFrame(np.array(X_nrm), columns=X.columns)

        # no need to handle the target class imbalance for clustering
        return X_std, X_nrm


    def get_selected_norm_features(self):
        return self.X_nrm_selected

    def get_selected_std_features(self):
        return self.X_std_selected

    def get_encoded_labels(self):
        return self.enc_y

    def get_multicollinearity_dict(self):
        return self.multicollinearity_dict

    def get_variance_inflation_factor_list(self):
        return self.variance_inflation_factor_list

    def get_target_collinearity_list(self):
        return self.target_collinearity_list

    def get_log_reg_dict(self):
        return self.log_reg_dict

    def get_selectKbest_dict(self):
        return self.selectKbest_dict

    def get_RFE_dict(self):
        return self.RFE_dict

    def get_reduced_selected_data(self):
        return self.pca_data_dict, self.tsne_data_dict, self.umap_data_dict




