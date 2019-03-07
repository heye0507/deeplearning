import numpy as np
import pandas as pd
import torch
import random


def preprocess(path,seed=47):
    df_label = pd.read_csv(path)
    df_split = df_label.copy()
    df_grouped = df_split.groupby('Id').apply(lambda x: x.sample(frac=0.2,random_state=seed))
    df_merged = pd.merge(left=df_split,right=df_grouped,on='Image',how='left',suffixes=('','_y'))
    df_merged['is_valid'] = df_merged.Id_y.isnull()!=True
    df_merged.drop('Id_y',axis=1,inplace=True)
    return df_merged


def mapk(preds,targs,k=5):
    batch_pred = preds.sort(descending=True)[1] #batch_size * classes
    return torch.tensor(np.mean([single_map(p,l,k) for l,p in zip(targs,batch_pred)]))

def single_map(pred,label,k=5):
    try:
        return 1/ ((pred[:k] == label).nonzero().item()+1)
    except ValueError:
        return 0.0