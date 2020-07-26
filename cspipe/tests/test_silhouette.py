import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.realpath('.'))
from cspipe.silhouette_coef import plotSampleWiseSilhouetteScore
import matplotlib.pyplot as plt
if __name__ == "__main__":

    metadata = pd.read_csv("./dataset/metadata.csv",index_col = 0)
    df = pd.read_csv("./dataset/feature_profile.csv",index_col=0)
    df = df.sort_index()
    metadata = metadata.sort_index()

    unIdenticalNum = np.sum(df.index != metadata.index)
    if unIdenticalNum !=0:
        raise IndexError("Unidentical index between feature and metadata dataframe.")
    fig,ax = plotSampleWiseSilhouetteScore(df,metadata,cluster_name="cluster",features_are_row=False,xlim=(-0.6,0.8),figsize = (5,10))
    #plt.show()
    fig.savefig("./cspipe/tests/test_sil_coef.pdf",bbox_inches = 'tight',dpi=800)

