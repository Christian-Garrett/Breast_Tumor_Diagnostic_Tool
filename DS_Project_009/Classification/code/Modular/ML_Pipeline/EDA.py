import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from random import sample
from numpy import random
from sklearn import preprocessing


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


def get_subplot_coords(curr_num, col_count):

    y = (curr_num // col_count)
    x = curr_num - (y * col_count)

    return y, x


def hopkins_test(df,m):

    d = len(df.columns) # columns
    n = len(df) # rows
    
    df = (df - df.min())/(df.max()-df.min()) *2 -1
    df = df / df.std()
    
    knn = NearestNeighbors(n_neighbors=2).fit(df.to_numpy())

    rand_df = pd.DataFrame(random.rand(m,d),index = range(0,m),columns=df.columns)
    rand_df = rand_df*2-1
    rand_df = rand_df * df.abs().max()

    ujd = []
    wjd = []
        
    for j in range(0, m):
        u_dist, _ = knn.kneighbors([rand_df.iloc[j]])
        ujd.append(u_dist[0][0])

        w_dist, _ = knn.kneighbors(df.sample(1))
        wjd.append(w_dist[0][1])

    return(sum(ujd) / (sum(ujd) + sum(wjd)))

def get_corr_mat(dataframe):

    corr_mat = dataframe.corr()
    corr_mat_slice = corr_mat[corr_mat.columns[:1]]

    return corr_mat_slice.iloc[1: , :]

def pairplot(dfx):

    if dfx.name == 'mean_df':
        x = "Mean"
    elif dfx.name == 'se_df':
        x = "Squared Error"
    elif dfx.name == 'worst_df':
        x = "Worst"

    sns.pairplot(data=dfx, hue='Diagnosis', palette='crest', corner=True).fig.suptitle('Pairplot for {} Features'.format(x), fontsize = 20)



class EDA:

    def __init__(self, dir_path, target, drop):

        self.dir_path = dir_path
        self.target = target
        self.df = pd.read_csv(dir_path)
        self.df.drop(columns=drop, inplace=True)
        self.tar_enc_df = self.encode_target()
        self.mean_cat_list, self.se_cat_list, self.worst_cat_list = self.get_feature_categories()
        self.df_mean, self.df_se, self.df_worst = self.categorize_feature_data()
        # self.df_mean_enc_std, self.df_se_enc_std, self.df_worst_enc_std = self.split_encoded_data('std')
        # self.df_mean_enc_nrm, self.df_se_enc_nrm, self.df_worst_enc_nrm = self.split_encoded_data('nrm')

        print('\n')
        self.mean_low_min, self.mean_high_min, self.mean_high_max = self.get_boxplot_bounds('mean')
        self.se_low_min, self.se_high_min, self.se_high_max = self.get_boxplot_bounds('se')
        self.worst_low_min, self.worst_high_min, self.worst_high_max = self.get_boxplot_bounds('worst')



    def encode_target(self):

        X = self.df.drop(columns=[self.target]).copy()
        '''
        # standardize the feature data
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X)
        X_std_features_df = pd.DataFrame(np.array(X_std), columns=X.columns)

        # normalize the feature data
        X_nrm = normalize(X)
        X_nrm_features_df = pd.DataFrame(np.array(X_nrm), columns=X.columns)
        '''

        # label encode the target variable
        le = preprocessing.LabelEncoder()

        target_enc_df = self.df.copy()
        target_enc_df.Diagnosis = le.fit_transform(self.df.Diagnosis)

        '''
        X_std_target_df = pd.DataFrame()
        X_nrm_target_df = pd.DataFrame()
        # concatenate the encoded target variable and feature dataframes together
        X_std_target_df['Diagnosis'] = le.fit_transform(self.df.Diagnosis) 
        X_nrm_target_df = X_std_target_df
        result_std_df = pd.concat([X_std_target_df, X_std_features_df], axis=1)
        result_nrm_df = pd.concat([X_nrm_target_df, X_nrm_features_df], axis=1)
        '''

        return target_enc_df


    def categorize_feature_data(self):

        df_mean = self.df[self.mean_cat_list] 
        df_se = self.df[self.se_cat_list]
        df_worst = self.df[self.worst_cat_list]

        return df_mean, df_se, df_worst

    '''
    def split_encoded_data(self, type):

        delimeter = ", "

        mean_string = self.target + ", " + delimeter.join(self.mean_cat_list)
        mean_list = list(mean_string.split(", "))

        se_string = self.target + ", " + delimeter.join(self.se_cat_list)
        se_list = list(se_string.split(", "))

        worst_string = self.target + ", " + delimeter.join(self.worst_cat_list)
        worst_list = list(worst_string.split(", "))

        if(type == 'std'):

            df_mean_enc = self.df_enc_std[mean_list] 
            df_se_enc = self.df_enc_std[se_list]
            df_worst_enc = self.df_enc_std[worst_list]

        elif(type == 'nrm'):
            df_mean_enc = self.df_enc_nrm[mean_list] 
            df_se_enc = self.df_enc_nrm[se_list]
            df_worst_enc = self.df_enc_nrm[worst_list]

        return df_mean_enc, df_se_enc, df_worst_enc
    '''

    def output_feature_category_boxplots(self):

        # save feature category boxplot info
        output = 'Classification\code\Modular\Output\Data_Exploration\Mean_Boxplot.png'
        plt.figure(figsize=(12,8))
        boxplot = self.df.boxplot(column=self.mean_cat_list)
        plt.xticks(rotation=45)
        plt.ylim(-5, 180) # todo: adjust these values automatically
        plt.savefig(output)
        plt.clf()

        output = 'Classification\code\Modular\Output\Data_Exploration\Squared_Error_Boxplot.png'
        plt.figure(figsize=(12,8))
        boxplot = self.df.boxplot(column=self.se_cat_list)
        plt.xticks(rotation=45)
        plt.ylim(-5, 180) # todo: adjust these values automatically
        plt.savefig(output)
        plt.clf()

        output = 'Classification\code\Modular\Output\Data_Exploration\Worst_Boxplot.png'
        plt.figure(figsize=(12,8))
        boxplot = self.df.boxplot(column=self.worst_cat_list)
        plt.xticks(rotation=45)
        plt.ylim(-5, 180) # todo: adjust these values automatically
        plt.savefig(output)
        plt.clf()


    def output_feature_category_histograms(self):

        sub_rows = 2
        sub_cols = 5
        # output histogram data by feature categories
        output = 'Classification/code/Modular/Output/Data_Exploration/Mean_Histograms.png'
        plt.style.use("bmh")
        fig, axarr = plt.subplots(sub_rows, sub_cols)
        for feature in self.mean_cat_list:
            row, col = get_subplot_coords(self.mean_cat_list.index(feature), sub_cols)
            axarr[row][col].hist(self.df[feature])
            axarr[row][col].set_title(feature, fontsize=8)
            axarr[row][col].tick_params(labelrotation=45)
        fig.tight_layout()
        plt.savefig(output)
        plt.clf()

        output = 'Classification/code/Modular/Output/Data_Exploration/Squared_Error_Histograms.png'
        plt.style.use("bmh")
        fig, axarr = plt.subplots(sub_rows, sub_cols)
        for feature in self.se_cat_list:
            row, col = get_subplot_coords(self.se_cat_list.index(feature), sub_cols)
            axarr[row][col].hist(self.df[feature])
            axarr[row][col].set_title(feature, fontsize=8)
            axarr[row][col].tick_params(labelrotation=45)
        fig.tight_layout()
        plt.savefig(output)
        plt.clf()

        output = 'Classification/code/Modular/Output/Data_Exploration/Worst_Histograms.png'
        plt.style.use("bmh")
        fig, axarr = plt.subplots(sub_rows, sub_cols)
        for feature in self.worst_cat_list:
            row, col = get_subplot_coords(self.worst_cat_list.index(feature), sub_cols)
            axarr[row][col].hist(self.df[feature])
            axarr[row][col].set_title(feature, fontsize=8)
            axarr[row][col].tick_params(labelrotation=45)
        fig.tight_layout()
        plt.savefig(output)
        plt.clf()


    def output_target_value_counts(self):
        
        # Plot the diagnosis value counts
        output = 'Classification/code/Modular/Output/Data_Exploration/Diagnosis_Value_Counts.png'
        self.df.Diagnosis.value_counts().plot(kind="bar", width=0.1, color=["lightgreen", "cornflowerblue"], legend=1, figsize=(8, 5))
        plt.xlabel("(0 = Benign) (1 = Malignant)", fontsize=12)
        plt.ylabel("Count", fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(["Benign"], fontsize=12)
        plt.savefig(output)
        plt.clf()


    def output_feature_correlation_matrices(self):

        delimeter = ", "

        mean_string = self.target + ", " + delimeter.join(self.mean_cat_list)
        mean_list = list(mean_string.split(", "))
        mean_df = self.tar_enc_df[mean_list]

        se_string = self.target + ", " + delimeter.join(self.se_cat_list)
        se_list = list(se_string.split(", ")) 
        se_df = self.tar_enc_df[se_list]        

        worst_string = self.target + ", " + delimeter.join(self.worst_cat_list)
        worst_list = list(worst_string.split(", ")) 
        worst_df = self.tar_enc_df[worst_list]                

        # Plot correlation matrix between the diagnosis and the mean features
        mean_features_df = get_corr_mat(mean_df)
        output = 'Classification/code/Modular/Output/Data_Exploration/Mean_Features_Correlations.png'
        plt.figure(figsize=(20, 8))
        plt.rcParams.update({'font.size': 6})
        plt.tight_layout()
        mean_features_df.plot(kind='bar', grid=True, title="Correlation of Mean Features with Diagnosis", color="cornflowerblue", rot=35)
        plt.savefig(output)
        plt.clf()

        # Plot correlation matrix between the diagnosis and the squared error features
        standard_error_df = get_corr_mat(se_df)
        output = 'Classification/code/Modular/Output/Data_Exploration/Squared_Error_Correlations.png'
        plt.figure(figsize=(20, 8))
        plt.rcParams.update({'font.size': 6})
        standard_error_df.plot(kind='bar', grid=True, title="Correlation of SE Features with Diagnosis", color="cornflowerblue", rot=35)
        plt.savefig(output)
        plt.clf()

        # Plot correlation matrix between the diagnosis and the worst features
        worst_features_df = get_corr_mat(worst_df)
        output = 'Classification/code/Modular/Output/Data_Exploration/Worst_Features_Correlations.png'
        plt.figure(figsize=(20, 8))
        plt.rcParams.update({'font.size': 6})
        worst_features_df.plot(kind='bar', grid=True, title="Correlation of Worst Features with Diagnosis", color="cornflowerblue", rot=35)
        plt.savefig(output)
        plt.clf()

        # Create a correlation matrrix
        corr_matrix = self.tar_enc_df.corr() 

        # Create a heatmap mask
        mask = np.zeros_like(corr_matrix, dtype=np.bool)
        mask[np.triu_indices_from(corr_matrix)] = True

        # Create a correlation matrix heatmap for all features
        output = 'Classification/code/Modular/Output/Data_Exploration/Corr_Heatmap.png'
        fig, ax = plt.subplots(figsize=(22, 10))
        ax = sns.heatmap(corr_matrix, mask=mask, annot=True, linewidths=0.5, fmt=".2f", cmap="YlGn");
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom + 0.5, top - 0.5);
        ax.set_title("Correlation Matrix Heatmap including all features")
        plt.savefig(output)
        plt.clf()


        # Mean features multicollinarity plot
        output = 'Classification/code/Modular/Output/Data_Exploration/Mean_Features_Multicollinearity.png'
        mean_df.name = "mean_df"
        pairplot(mean_df)
        plt.savefig(output)
        plt.clf()

        # Squared error features multicollinarity plot
        output = 'Classification/code/Modular/Output/Data_Exploration/Squared_Error_Multicollinearity.png'
        se_df.name = "se_df"
        pairplot(se_df)
        plt.savefig(output)
        plt.clf()

        # Worst features multicollinarity plot
        output = 'Classification/code/Modular/Output/Data_Exploration/Worst_Features_Multicollinearity.png'
        worst_df.name = "worst_df"
        pairplot(worst_df)
        plt.savefig(output)
        plt.clf()


    def output_feature_category_distribution_comparisons(self):

        # Create separate data frames for the malignant and benign data
        dfM = self.df[self.df[self.target] == 'M']
        dfB = self.df[self.df[self.target] == 'B']

        # Plot the 'mean' distribution vs diagnosis data
        output = 'Classification/code/Modular/Output/Data_Exploration/Mean_vs_Diagnosis_Distributions.png'
        plt.rcParams.update({'font.size': 8})
        fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(8, 10))
        axes = axes.ravel()
        for idx, ax in enumerate(axes):
            ax.figure
            binwidth = (max(self.df[self.mean_cat_list[idx]]) - min(self.df[self.mean_cat_list[idx]])) / 50
            ax.hist([dfM[self.mean_cat_list[idx]], dfB[self.mean_cat_list[idx]]],
                    bins=np.arange(min(self.df[self.mean_cat_list[idx]]), max(self.df[self.mean_cat_list[idx]]) + binwidth, binwidth), alpha=0.5,
                    stacked=True, label=['M', 'B'], color=['b', 'g'])
            ax.legend(loc='upper right')
            ax.set_title(self.mean_cat_list[idx])
        plt.tight_layout()
        plt.savefig(output)
        plt.clf()

        # Plot the 'squared error' distribution vs diagnosis data
        output = 'Classification/code/Modular/Output/Data_Exploration/Squared_Error_vs_Diagnosis_Distributions.png'
        plt.rcParams.update({'font.size': 8})
        fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(8, 10))
        axes = axes.ravel()
        for idx, ax in enumerate(axes):
            ax.figure
            binwidth = (max(self.df[self.se_cat_list[idx]]) - min(self.df[self.se_cat_list[idx]])) / 50
            ax.hist([dfM[self.se_cat_list[idx]], dfB[self.se_cat_list[idx]]],
                    bins=np.arange(min(self.df[self.se_cat_list[idx]]), max(self.df[self.se_cat_list[idx]]) + binwidth, binwidth), alpha=0.5,
                    stacked=True, label=['M', 'B'], color=['b', 'g'])
            ax.legend(loc='upper right')
            ax.set_title(self.se_cat_list[idx])
        plt.tight_layout()
        plt.savefig(output)
        plt.clf()

        # Plot the 'worst features' distribution vs diagnosis data
        output = 'Classification/code/Modular/Output/Data_Exploration/Worst_Features_vs_Diagnosis_Distributions.png'
        plt.rcParams.update({'font.size': 8})
        fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(8, 10))
        axes = axes.ravel()
        for idx, ax in enumerate(axes):
            ax.figure
            binwidth = (max(self.df[self.worst_cat_list[idx]]) - min(self.df[self.worst_cat_list[idx]])) / 50
            ax.hist([dfM[self.worst_cat_list[idx]], dfB[self.worst_cat_list[idx]]],
                    bins=np.arange(min(self.df[self.worst_cat_list[idx]]), max(self.df[self.worst_cat_list[idx]]) + binwidth, binwidth), alpha=0.5,
                    stacked=True, label=['M', 'B'], color=['b', 'g'])
            ax.legend(loc='upper right')
            ax.set_title(self.worst_cat_list[idx])
        plt.tight_layout()
        plt.savefig(output)
        plt.clf()


    def get_feature_categories(self):

        # create 'feature category' lists
        feat_names = list(self.df.columns)
        feat_list = ""
        for name in feat_names:
            feat_list = (feat_list + name + ", ")

        mean_matches = re.findall("Mean\s\w+\s?\w+", feat_list)
        worst_matches = re.findall("Worst\s\w+\s?\w+", feat_list)
        se_matches = re.findall("\w+?\s?\w+\sSE", feat_list)

        return mean_matches, se_matches, worst_matches


    def get_boxplot_bounds(self, type):

        # get range info for boxplots
        desc_df = self.df.describe().T

        col_start = 0
        if(type == 'mean'):
            col_start = 9
        elif(type == 'se'):
            col_start = 19
        elif(type == 'worst'):
            col_start = 29
        col_end = col_start + 9

        # get the corresponding boxplot boundary values
        low_min = float(desc_df.iloc[col_start:col_end,[3]].min())
        high_min = float(desc_df.iloc[col_start:col_end,[3]].max())
        high_max = float(desc_df.iloc[col_start:col_end,[1]].max())

        print(f'{type}: low min is {low_min}, high min is {high_min}, high max is {high_max}')

        return low_min, high_min, high_max
    


    def explore(self):

        # print the dataframe info
        print(self.df.info())

        print("\nTotal Null Values:", np.sum(self.df.isnull().sum()))
        print("Total Missing Values:", np.sum(self.df.isna().sum()))

        print('DataFrame Tail (5):')
        print(self.df.tail())

        print('\n')

        print('DataFrame Describe:')
        print(self.df.describe())

        # determine data "cluster-ability" with the hopkins score
        m = 10
        X = self.df.drop(columns=[self.target])
        hop_score = hopkins_test(X, m)
        print(f'\nHopkins Score: {hop_score}\n')

        self.output_target_value_counts()

        self.output_feature_category_boxplots()

        self.output_feature_category_histograms()

        self.output_feature_category_distribution_comparisons()

        self.output_feature_correlation_matrices()


    def get_df_data(self):
        return self.df, self.tar_enc_df

    def get_feature_category_lists(self):
        return self.mean_cat_list, self.se_cat_list, self.worst_cat_list


