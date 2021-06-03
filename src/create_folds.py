import config
import pandas as pd
from sklearn.model_selection import StratifiedKFold

if __name__ == '__main__':
    df = pd.read_csv(config.TRAIN_DATA)
    df.loc[:, "kfold"] = -1
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    targets = df["Is_Lead"].values

    skf = StratifiedKFold(n_splits=5)

    for fold,(trn, val) in enumerate(skf.split(X=df,y=targets)):
        df.loc[val,"kfold"] = fold

    df.to_csv(config.TRAIN_FOLDS, index=False)