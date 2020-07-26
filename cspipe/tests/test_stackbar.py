import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.realpath('.'))

if __name__ == "__main__":
    script = ["python ./cspipe/stackbar.py "]
    metadata = ["-m ./dataset/metadata.csv "]
    df = ["-f ./dataset/feature_profile.csv "]
    colnames = ["-n cluster "]
    far = ["-far False "]
    output = ["-o","./tests/stackbar.pdf"]
    cmd =  ''.join(script + metadata + df + colnames +far +output)
    print(cmd)
    os.popen(cmd)
    print("done")