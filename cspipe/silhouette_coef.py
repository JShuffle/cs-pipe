from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import ticker
from matplotlib.colors import to_hex
import numpy as np
import pandas as pd
import os
import argparse
from .utils import str2bool,setColorConf
USAGE="""
XXX.
"""
def parser():
    args = argparse.ArgumentParser(description=USAGE)
    args.add_argument('-f','--featureDataFrame')
    args.add_argument('-m','--metadataDataFrame')
    args.add_argument('-x','--metric')
    args.add_argument('-n','--clustername')
    args.add_argument('-far','--feature_are_row',default="True",type=str2bool)
    args.add_argument('-o','--output',default = './SilhouetteScore.pdf')
    args = args.parse_args()
    return args

def checkIndexIdenticcal(featureDataFrame,metadataDataFrame):
    return np.sum(featureDataFrame.index != metadataDataFrame.index)

def plotSampleWiseSilhouetteScore(featureDataFrame,
                                  metadataDataFrame,
                                  cluster_name,
                                  features_are_row=True,
                                  n_clusters = None,
                                  metric = "euclidean",
                                  order=None,
                                  colors="tab20",
                                  xlim=(-1,1),
                                  y_lower = 10,
                                  interval=15,
                                  figsize = (5,10),**kwargs):
    """
    featureDataFrame(Required):,pd.DataFrame
        row per feature(taxonomy),col per sample. If setting metric = "precomputed",
        then featureDataFrame should be distance dataframe precomputed.

    metadataDataFrame(Required): pd.DataFrame
        metadata containing group information(make sure the index same with featureDataFrame: feature).

    cluster_name(Required):
        column name of cluster in the metadataDataFrame.

    n_clusters(Required):

    order:list
        the plot order of each cluster.

    metric:str
        default = "precomputed".
        If metric is a string, it must be one of the options allowed by sklearn.metrics.pairwise.pairwise_distances.
        If using "precomputed" as the metric, then the featureDataFrame should be the distance dataframe itself.
        Precomputed distance matrices must have 0 along the diagonal.

    color:str or list.
        color of each cluster.
        If passing str, then calling matplotlib inner palette.
        If passing list, then using the list color directly.

    """
    # average silhouette_score
    if features_are_row:
        featureDataFrame = featureDataFrame.T.sort_index()
    else:
        featureDataFrame = featureDataFrame.sort_index()
    metadataDataFrame = metadataDataFrame.sort_index()

    unIdenticalNum = checkIndexIdenticcal(featureDataFrame,metadataDataFrame)
    if unIdenticalNum !=0:
        raise IndexError("Exist un-identical index between 2 dataframe.")

    X = featureDataFrame.values
    if cluster_name not in metadataDataFrame.columns:
        raise ValueError(cluster_name + " not exist in metadata.")

    cluster_labels = metadataDataFrame.loc[:,cluster_name].values
    unique_labels = np.unique(cluster_labels)

    if not n_clusters:
        n_clusters = len(unique_labels)

    if metric =="precomputed":
        np.fill_diagonal(X, 0)

    silhouette_avg = silhouette_score(X, cluster_labels,metric=metric)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    # visualization
    plt.rc('font', family='Arial')
    fig = plt.figure(1, figsize=figsize)
    ax = fig.add_subplot(1,1,1)
    ax.set_xlim([xlim[0], xlim[1]])  

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels,metric=metric)

    y_lower = y_lower

    if not order: # if not specified.
        order = np.sort(unique_labels)
    if not set(order).issubset(set(unique_labels)):
        raise ValueError("provided order does not match the exact cluster label.")

    # color
    # if specified by user(passing a list).
    if isinstance(colors,list):
        colors_list = colors
        ## TODO:raise warning if len of list not consistent with ngroups.
    
    # if using matplotlib inner palette
    elif isinstance(colors,str):
        if colors:
            colors_list = setColorConf(colors=colors,ngroups = n_clusters)
        else:
            colors_list = None

    for i,cs in enumerate(order):
        # Aggregate the silhouette scores for samples belonging to cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == cs]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]

        y_upper = y_lower + size_cluster_i

        #color = cm.nipy_spectral(float(i) / n_clusters)
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                        0, # start
                        ith_cluster_silhouette_values, # end
                        facecolor=colors_list[i],
                        edgecolor=colors_list[i],
                        alpha=0.7,**kwargs)

        # Label the silhouette plots with their cluster numbers at the middle
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(cs))

        # Compute the new y_lower for next plot
        y_lower = y_upper + interval

    #ax1.set_title("The silhouette plot")
    ax.set_title("Silhouette Coefficient",size=15)
    #ax.set_xlabel("Silhouette Coefficient",size=15)
    #ax.set_ylabel("Cluster",size=15,rotation="horizontal")
    # The vertical line for average silhouette score of all the values
    #ax.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax.set_yticks([])  # Clear the yaxis labels / ticks
    #ax.set_xticks([-0., 0, 0.2, 0.4, 0.6, 0.8, 1])
    ## set spines invisible
    ax.spines['left'].set_color(None)
    ax.spines['right'].set_color(None)
    ax.spines['top'].set_color(None)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.2))
    for i in ax.xaxis.get_ticklines():
        i.set_linewidth(1.5)
        i.set_markeredgewidth(1.5)
    for i in ax.xaxis.get_ticklabels():
        i.set_fontsize(13)
        #i.set_fontfamily('serif')

    #plt.show()
    return fig,ax
if __name__ == "__main__":
    args = parser()
    df = args.featureDataFrame
    metadata = args.metadataDataFrame
    metric = args.metric
    far = args.feature_are_row
    
    try:
        if ".csv" in df:
            df = pd.read_csv(df,index_col=0)
        elif ".txt" in df:
            df = pd.read_csv(df,index_col=0,sep="\t")
        elif ".tsv" in df:
            df = pd.read_csv(df,index_col=0,sep="\t")
        elif ".xlsx" in df:
            df = pd.read_excel(df,index_col=0)
    except:
        print("Unrecognized format, please check your feature dataframe.")

    try:
        if ".csv" in metadata:
            metadata = pd.read_csv(metadata,index_col=0)
        elif ".txt" in metadata:
            metadata = pd.read_csv(metadata,index_col=0,sep="\t")
        elif ".tsv" in metadata:
            metadata = pd.read_csv(metadata,index_col=0,sep="\t")
        elif ".xlsx" in metadata:
            metadata = pd.read_excel(metadata,index_col=0)
    except:
        print("Unrecognized format, please check your metadata dataframe.")

    cs = args.clustername
    fname = args.output
    fig = plotSampleWiseSilhouetteScore(df,metadata,cluster_name=cs,metric = metric,features_are_row=far)
    plt.savefig(fname)

