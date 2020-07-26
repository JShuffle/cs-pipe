import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import to_hex
import seaborn as sns
import pandas as pd
import numpy as np
import argparse
import os
import sys
from .utils import str2bool,setColorConf

USAGE="""
Desc:
script to draw stacked violinplot
Example:
python stackviolin.py \
    -f profile.csv \
    -m cluster.csv \
    -n cluster \
    -far False \
    -F 0.05 \
    -o svl.pdf
"""


class stackViolin():
    def __init__(self,
                featureDataFrame:pd.DataFrame,
                groupDataFrame:pd.DataFrame):
        """
        featureDataFrame:
            feature table with one feature-id(e.g. taxa1,taxa2) per columns and one row per sample-id.
        groupDataFrame:
            group information of sample-id, containing 2 columns: sample-id and its group info.
        """
        self.configure = {}
        if np.sum(featureDataFrame.index != groupDataFrame.index) !=0:
            raise IndexError("Unidentical index between featureDataFrame and groupDataFrame")
        self.violinDataFrame = self.createViolinDataFrame(featureDataFrame,groupDataFrame)

    def getConfig(self,kwargs):
        return self.configure[kwargs]

    def createViolinDataFrame(self,featureDataFrame,groupDataFrame):
        """
        each row a sample.
        new column names:sample-id/group/feature1/featur2/.../featuren

        """
        violinDataFrame = featureDataFrame.join(groupDataFrame)
        self.configure['violinDataFrame'] = violinDataFrame
        return violinDataFrame

    def setAxesSize(self,fig,ith_axes,axes_unit_height):
        """
        ith_axes: the order of axes(denote by ith).
        """
        # first axes
        if ith_axes==0:
            size = [0,0,1,axes_unit_height]
            ax = fig.add_axes(size) # left start, bottom start ,right end ,top end
        else:
            #print(0,axes_unit_height*ith_axes,1,axes_unit_height)
            size = [0,np.round(axes_unit_height*ith_axes,2),1,axes_unit_height]
            ax = fig.add_axes(size)
        self.configure['axes_'+str(ith_axes)+'_size'] = size
        return ax

    def setTickConf(self,ax,ith_axes,ylabel):
        ax.tick_params(
            axis='y',
            left=False,
            right=True,
            labelright=True,
            labelleft=False,
            labelsize=12,
            length=1,
            pad=1,)

        ax.set_ylabel(
            ylabel,
            rotation=0,
            fontsize=13,
            labelpad=8,
            ha='right',
            va='center',)
        #ax.set_ylim(0,1)
        ax.set_xlabel('')
        if ith_axes > 0:
            ax.xaxis.set_major_locator(ticker.NullLocator())


    def setRotation(self,ax,xrotate):
        """
        xrotate:degree of xcord label rotation.
        """
        ## TODO:yrotate
        if isinstance(xrotate,int):
            for i in ax.xaxis.get_ticklabels():
                i.set_rotation(xrotate)

    def plotViolin(self,
                    figsize,
                    order = None,
                    axes_unit_height=0.3,
                    orient="h",
                    color_palette="tab20",
                    xrotate = 0,
                    **kwargs):

        # get unique group from the last column(group info).
        group_name = self.violinDataFrame.columns[-1]
        groups = np.unique(self.violinDataFrame.iloc[:,-1].values)
        ngroups = len(groups)
        self.configure['groups'] = groups
        self.configure['n_groups'] = ngroups

        # get features from column names.
        features = self.violinDataFrame.columns[:-1]
        self.configure['features'] = features


        if not order:
            order = np.sort(groups)
        else:
            # check whether the set of provided order identical with groups.
            if set(order) != set(groups):
                raise ValueError("Unidentical values between provided order and groups.")

        fig = plt.figure(figsize=figsize)
        self.configure['figsize'] = figsize

        # if specified by user(passing a list).
        if isinstance(color_palette,list):
            colors_list = color_palette
            ## TODO:raise warning if len of list not consistent with ngroups.
        # if using matplotlib inner palette
        elif isinstance(color_palette,str):
            if color_palette:
                colors_list = setColorConf(colors=color_palette,ngroups = ngroups)
            else:
                colors_list = None
        self.configure['color_palette'] = colors_list

        # for loop adding stacked violin plot
        for i,feature in enumerate(features):
            ## TODO:orient
            ax = self.setAxesSize(fig,ith_axes=i,axes_unit_height=axes_unit_height)
            ax.grid(False)
            sns.violinplot(x=group_name,
                        y=feature,
                        data=self.violinDataFrame,
                        split=True, inner=None,
                        scale='width',
                        ax=ax,
                        palette=colors_list,
                        order=order,
                        **kwargs)

            self.setTickConf(ax,ith_axes=i,ylabel = feature)


        if xrotate == 0:rotate=False
        else:rotate=True

        # if rotate x labels
        if rotate:
            ax0 = fig.axes[0]
            self.setRotation(ax0,xrotate = xrotate)

        sns.set_style('ticks')
        return fig,ax

if __name__ == "__main__":
    args = argparse.ArgumentParser(description=USAGE)
    args.add_argument('-f','--featureDataFrame',help="dataframe contains feature(taxonomy/gene/others) and sample-id.")
    args.add_argument('-m','--metadataDataFrame',help="dataframe contains cluster information and sample-id.")
    args.add_argument('-n','--clustername',help="the column names of which containing cluster information.")
    args.add_argument('-far','--feature_are_row',default="False",type=str2bool,help="set True if feature are row else False.")
    args.add_argument('-F','--filter',default="0",
                        help="min threshold to filter the feature(by its mean abundance in each cluster).If 0,none feature would be filtered.")
    args.add_argument('-c','--cmap',default = "tab20",help ="color palette of clusters.")
    args.add_argument('-s','--figsize',default = "10,5",help ="size of figure(split by comma).")
    args.add_argument('-o','--output',default = './stackViolin.pdf',help ="path to save the violinplot.")

    args = args.parse_args()

    df = args.featureDataFrame
    metadata = args.metadataDataFrame
    csname = args.clustername
    far = args.feature_are_row
    fthres = args.filter
    
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

    if far: # feature are row?
        df = df.T

    df = df.sort_index()
    metadata = metadata.sort_index()

    unIdenticalNum = np.sum(df.index != metadata.index)
    if unIdenticalNum !=0:
        raise IndexError("Unidentical index between feature and metadata dataframe.")

    ## sort by total abundance of feature
    sorted_feature = df.sum().sort_values(ascending=False).index 
    sorted_f_list = list(sorted_feature)
    df = df.loc[:,sorted_f_list]

    if fthres != "0":
        new_df = df.join(metadata)
        mean_abd = new_df.groupby(csname).mean()
        bool_mean_abd = mean_abd > float(fthres)
        bool_mean_abd = bool_mean_abd.sum() !=0  ## mean abundance gthan fthres in at least one clusters
        bool_mean_abd_gthan_thres_percent = bool_mean_abd[bool_mean_abd]
        df = df.loc[:,bool_mean_abd_gthan_thres_percent.index]

    sv = stackViolin(featureDataFrame=df,groupDataFrame=metadata.loc[:,csname])
    #print(sv.violinDataFrame)
    fname = args.output
    fsize = args.figsize
    color = args.cmap
    fs1,fs2 = int(fsize.split(",")[0]),int(fsize.split(",")[1])

    fig,ax = sv.plotViolin(axes_unit_height=0.1,
                        figsize=(fs1, fs2),
                        color_palette = color)
    fig.savefig(fname,bbox_inches = 'tight',dpi=1000)
