import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


# Globals
TARGET = 'Diagnosis'
COLOURS = ["r","g"]
LABELS = ["Benign","Malignant"]
NUM_LABELS = len(LABELS)




# create the crosstab color (label) dictionary
def create_color_dict(num_clust, offset):
    
    color_dict = {}
    for color in COLOURS:
        color_dict.update({color:{}})
    for key in color_dict.keys():
        [color_dict[key].update({i-offset:0}) for i in range(num_clust)]

    return color_dict


def plot_dim_preds(dim_df, X_dim, labels, prefix, mod_name, noise_offset):
 
    unique_vals = len(set(dim_df['clust2'].values))
    # Create a dictionary to store 2D reduced target labels/prediction counts
    label_preds_dict = create_color_dict(unique_vals, noise_offset)

    # Plot label predictions as numbers with actual labels as colors
    plt.figure(figsize=(10,5))
    for i in range(X_dim.shape[0]):
        plt.text(x=X_dim[i, 0], y=X_dim[i, 1], s=str(dim_df[f'clust{NUM_LABELS}'][i])
                 , color=COLOURS[int(labels[i])], fontdict={'weight': 'bold', 'size': 8})
        # Increment the label dictionary prediction class counter
        label_preds_dict[COLOURS[int(labels[i])]][dim_df[f'clust{NUM_LABELS}'][i]] += 1

    # Add the counter dictionary to the plot
    classes = ''
    for i in range(NUM_LABELS):
        classes = classes + f'{COLOURS[i]} : {label_preds_dict[COLOURS[i]]} '

    output = f'Classification\code\Modular\Output\Model_Analysis\models\{mod_name}\dim_reduction\{prefix}\{mod_name}_{prefix}_{NUM_LABELS}_cluster_full_visualization.png'
    plt.xticks([])
    plt.yticks([])
    mins = np.min(X_dim, axis=0)
    maxs = np.max(X_dim, axis=0)
    plt.xlim([mins[0], maxs[0]])
    plt.ylim([mins[1], maxs[1]])
    plt.axis('off')
    plt.legend(title=classes, prop={'size': 8})
    plt.savefig(output)
    plt.clf()


# calculate the classification report accuracy score
def get_accuracy(scoring):
    
    run_count = 0
    for key in scoring.keys():
        run_count += scoring[key]
    run_count /= (len(scoring))

    return run_count


# calculate the precision, recall and f1 scores
def create_score_matrix(color_matrix):

    # index positions for each label
    cols = color_matrix.columns.values.tolist()
    index_dict = {}
    for i in range(len(cols)):
        index_dict.update({i:cols[i]})

    # precision score divisors for each label
    precision_divisor = {}
    fp = 0
    start_pos = 0
    for row_iter in range(len(color_matrix)):
        curr_loc = cols.index(index_dict[row_iter])
        for col_iter in range(len(cols)):
            fp += color_matrix[index_dict[col_iter]].iloc[curr_loc]
        precision_divisor.update({cols[row_iter]:fp})
        fp = 0
        start_pos += 1

    # recall score divisors for each label
    recall_divisor = {}
    fn = 0
    start_pos = 0
    for col_iter in range(len(cols)):  
        for row_iter in range(len(color_matrix)):
            curr_loc = cols.index(index_dict[row_iter])
            fn += color_matrix[index_dict[col_iter]].iloc[curr_loc]
        recall_divisor.update({cols[col_iter]:fn})
        fn = 0
        start_pos += 1

    true_positives = {}
    for i in range(len(cols)):
        true_positives.update({cols[i]:color_matrix[cols[i]].iloc[i]})

    precision_score = {}
    cols = color_matrix.columns
    for i in range(len(cols)):
        precision_score.update({cols[i]:true_positives[index_dict[i]]/precision_divisor[index_dict[i]]})

    recall_score = {}
    cols = color_matrix.columns
    for i in range(len(cols)):
        recall_score.update({cols[i]:true_positives[index_dict[i]]/recall_divisor[index_dict[i]]})

    f1_score = {}
    for col in cols:
        f1 = 2 * (precision_score[col] * recall_score[col]) / (precision_score[col] + recall_score[col])
        f1_score.update({col:f1})

    accuracy_score = get_accuracy(f1_score)
    matrix = pd.DataFrame([precision_score, recall_score, f1_score], index = ['Precision Score', 'Recall Score', 'F1 Score'])

    return accuracy_score, matrix


def create_pred_dict(dim_df, labels, offset):

    # Create a dictionary to store label predictions for 2D visualizations
    label_preds_dict = create_color_dict(NUM_LABELS, COLOURS, offset)

    for i in range(len(dim_df)):
        # increment the label dictionary prediction class counter
        actual = labels[i]
        clr = COLOURS[int(labels[i])]
        pred = dim_df[f'clust{str(NUM_LABELS)}'][i]
        label_preds_dict[COLOURS[int(labels[i])]][dim_df[f'clust{NUM_LABELS}'][i]] += 1

    return label_preds_dict


# create a confusion matrix regardless of the number of clusters assigned
def create_conf_mat(col_dict):

    Clusters = list(list(col_dict.values())[0].keys())
    Colors = list(col_dict.keys())

    val_dict = {}
    for label, name in zip(Colors, LABELS):
        val_dict[name] = list(col_dict[label].values())

    df = pd.DataFrame(data=val_dict, index=Clusters)

    transposed_df = df.T
    trans_max = transposed_df.idxmax()

    Label_Dict = {}
    for label, name in zip(Colors, LABELS):
        index = Colors.index(label)
        Label_Dict.update({name:index})

    Inv_Label_Dict = {v: k for k, v in Label_Dict.items()}

    row_list = []
    for ele in trans_max:
        row_list.append(Label_Dict[ele])

    dat_array = np.empty(len(Colors), dtype=object)
    dat_array[...] = [np.array([]) for _ in range(dat_array.shape[0])]

    split_data = df.to_dict('split')
    for i,j in zip(range(len(split_data['data'])), row_list):
        if not dat_array[j].size == 0:
            dat_array[j] = dat_array[j] + np.array(split_data['data'][i])
        else:
            dat_array[j] = np.array(split_data['data'][i])

    matrix_dict = {}
    for i in range(len(LABELS)):
        matrix_dict.update({Label_Dict[LABELS[i]]:dat_array[i]})

    confusion_matrix = pd.DataFrame(data=matrix_dict).T
    confusion_matrix.rename(columns=Inv_Label_Dict, inplace=True)

    return confusion_matrix


def print_final_model_evaluation(mod_array, range, num_dims, reduced_feats, y, rtype):

    # Create a dataframe to store 6 predictions for 2-5 clusters
    preds_df = pd.DataFrame()

    # Add PCA predictions for a small range of clusters
    clust_array = np.arange(range[0], range[1], dtype=int)
    for nclust, mod in zip(clust_array, mod_array):
        pred = mod.fit_predict(reduced_feats)
        preds_df['clust' + str(nclust)] = pred

        print('\n')
        print('{} {} Cluster {} Dimension ARI Score: {}'.format(rtype, nclust, num_dims, metrics.adjusted_rand_score(y, pred)))
        print('{} {} Cluster {} Dimension Silhouette Score: {}'.format(rtype, nclust, num_dims, metrics.silhouette_score(reduced_feats, pred, metric='cosine')))


    pred_dict = create_pred_dict(preds_df, y, 0)
    conf_mat = create_conf_mat(pred_dict)
    print(f'\n{rtype} Final Model Confusion Matrix: \n{conf_mat}\n')

    # create a classification report that corresponds to the custom label ordering and cluster size created by the clustering algorithm  
    accuracy, score_matrix_df = create_score_matrix(conf_mat)
    print(f'{rtype} Final Model Score Matrix = \n{score_matrix_df}')
    print(f'{rtype} Final Model Accuracy Score = {accuracy}\n')