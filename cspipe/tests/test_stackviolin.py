import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.realpath('.'))
#print(os.path.realpath('.'))
from cspipe.stackviolin import stackViolin
import matplotlib.pyplot as plt
if __name__ == "__main__":
    metadata = pd.read_csv("./dataset/metadata.csv",index_col = 0)
    df = pd.read_csv("./dataset/feature_profile.csv",index_col=0)
    df = df.sort_index()
    metadata = metadata.sort_index()

    unIdenticalNum = np.sum(df.index != metadata.index)
    if unIdenticalNum !=0:
        raise IndexError("Unidentical index between feature and metadata dataframe.")

        
    sv = stackViolin(featureDataFrame=df,groupDataFrame=metadata.loc[:,"cluster"])
    fig,ax = sv.plotViolin(axes_unit_height=0.1,
                        figsize=(10, 5),
                        color_palette = "tab20")
    fig.savefig("./cspipe/tests/test_sv.pdf",bbox_inches = 'tight',dpi=800)

