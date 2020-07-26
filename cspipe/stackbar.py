import argparse
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.cm as cm
from matplotlib.colors import to_hex
#import sys
#import os
#sys.path.append(os.path.realpath('.'))
from utils import str2bool,setColorConf
USAGE="""scripts to draw cluster specific stack barplot."""

def resortFirstSample(df):
    """
    df: row feature,col sample.
    """
    # re-rank the order of sample by the max-abundance taxon in sample1
    firstsam = df.columns[0]
    firstsam = df.loc[:,firstsam]
    sort_index = firstsam.sort_values().index
    return df.loc[sort_index,:]
def axesConf(df,axes=None):
    """
    df: row feature,col sample.
    axes:fig,ax object
    """
    # basic config
    if not axes:
        plt.rc('font', size=15)
        plt.rc('font', serif='Helvetica Neue')
        fig, ax = plt.subplots(figsize=(20,5))
        ax.set_facecolor('w')
        #ax.yaxis.set_visible(False)
        ax.set_ylim(0, df.sum().max())
        ax.set_ylabel('Relative abundance',color ='black')
    else:
        fig,ax = axes
    return fig,ax

def topk(df,top=19):
    """
    df: row feature,col sample.
    """
    topk_feature = df.T.sum().sort_values(ascending=False)[:top].index
    topk_feature = list(topk_feature)
    topk_feature = topk_feature[::-1]
    df_top = df.loc[topk_feature,:]
    df_top.loc['others',:] = 1- df_top.sum()
    order = ['others'] + topk_feature
    df_top = df_top.loc[order,:]
    df_top = df_top.sort_values(df_top.index[-1],axis=1,ascending=False)
    return df_top

def storer(df,metadata,csname,top=19):
    profiles={}
    csset = list(set(metadata.loc[:,csname]))
    for i,cs in enumerate(csset):
        cur_cs_sam_id = metadata[metadata.loc[:,csname] == cs].index
        cur_df = df.loc[:,cur_cs_sam_id]
        cur_df = topk(cur_df,top=top)
        profiles[cs] = cur_df
    return profiles

def StackBarplot(df,
                 bin_width=0.1,
                 rank=False,axes=None,
                fontsize=15,
                 linewidth=0.1,
                yticklabel=True,save=False):
    """
    stacked barplot for basic visualization of taxonomic abundance df.

    df: pandas DataFrame.
        one col per sample, and one row per feature(taxonomy)
    """
    if rank:
        df = resortFirstSample(df)
    fig,ax = axesConf(df,axes=axes)
    # prefix data structure
    featureList = list(df.index)[::-1]

    # color
    colors = list(cm.tab20.colors)
    category_colors  = [to_hex(color) for color in colors]

    xrange = np.arange(0,len(df.columns))
    #xrange = np.arange(0,bin_width* len(df.columns),step=bin_width) ## todo:乘法有问题,容易使得xrange与df.columns的长度不一致
    starts= [0 for i in range(len(df.columns))]

    for (i,feature) in enumerate(featureList):
        # stacked barplot: add bar one by one sample
        ## color
        #category_colors = color_conf(len(taxonList))
        #category_colors = plt.get_cmap('tab20')(np.linspace(0.15, 0.85, len(taxonList)))

        ## stacked bar

        height = df.loc[feature,:].values
        height = np.array(height)
        ax.bar(xrange, height, bottom=starts, width=bin_width,
               linewidth=linewidth,
               edgecolor='black',
               align='edge',
                label=feature, color=category_colors[i])

        starts = [i+j for i,j in zip(starts,height)]

    ax.legend(bbox_to_anchor=(1, 0),
              loc='lower left',
              fontsize=fontsize,
              facecolor='w')
    ## tick setting
    for xline,yline in zip(ax.get_xticklines(),ax.get_yticklines()):
        xline.set_visible(False)
        #yline.set_visible(False)

    for (xlabel,ylabel) in zip(ax.get_xticklabels(),ax.get_yticklabels()):
        ylabel.set_color('black')
        ylabel.set_fontsize(10)


    ax.xaxis.set_major_locator(ticker.NullLocator())
    ## set spines invisible
    ax.spines['bottom'].set_color(None)
    ax.spines['right'].set_color(None)
    ax.spines['top'].set_color(None)
    #if save:
        #baseconf.BaseAxes.savefig('StackBarplot')
    plt.tight_layout()
    return fig,ax


if __name__ == "__main__":
    args = argparse.ArgumentParser(description=USAGE)
    args.add_argument('-f','--featureDataFrame')
    args.add_argument('-m','--metadataDataFrame')
    args.add_argument('-n','--clustername')
    args.add_argument('-far','--feature_are_row',default="False",type=str2bool)
    args.add_argument('-t','--top',default="19",help="maximum feature number to display")
    args.add_argument('-o','--output',default = './cs_barplot.pdf')
    args = args.parse_args()

    df = args.featureDataFrame
    metadata = args.metadataDataFrame
    csname = args.clustername
    far = args.feature_are_row
    top = args.top

    try:
        top = int(top)
    except TypeError:
        msg="Unknown top args."
        print(msg)
    fname = args.output
    
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

    if not far: # feature are row?
        df = df.T

    pfs = storer(df,metadata,csname=csname,top=top)

    plt.rc('font', size=20)
    plt.rc('font', family='Arial')

    csnum = len(pfs.keys())
    csset = list(pfs.keys())
    vsize =6*csnum
    fig = plt.figure(figsize = [15,vsize])

    for i,cs in enumerate(csset):
        ax = fig.add_subplot(csnum,1,i+1)
        if len(pfs[cs]) <30:
            line_width = 0.1
        else:
            line_width = 0
        fig,ax = StackBarplot(pfs[cs],
                        bin_width=1,
                        axes =(fig,ax),
                        xticklabel=False,
                        yticklabel=True,
                        fontsize=10,
                        linewidth = line_width)

        ax.set_title("cluster:%s" % cs)
    fig.savefig(fname,bbox_inches = 'tight',dpi=1000)
