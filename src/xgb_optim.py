import os
import config
import joblib
import optuna
import pandas as pd
import numpy as np
from pipeline import pipe0

from xgboost import XGBClassifier
from sklearn import metrics


def run_training(fold,model_name="model",param=None):
    # load the full training data with folds
    df = pd.read_csv(config.TRAIN_FOLDS)
    # all columns are features except id, target and kfold columns
    features = [
    f for f in df.columns if f not in ("ID", "Is_Lead", "kfold")
    ]

    # get training data using folds
    df_train = df[df.kfold != fold].reset_index(drop=True)
    # get validation data using folds
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    
    # fit ohe on training + validation features
    full_data = pd.concat([df_train[features], df_valid[features]],axis=0)

    # train valid
    xtrain = df_train[features]
    ytrain = df_train[config.TARGET].values
    
    xvalid = df_valid[features]
    yvalid = df_valid[config.TARGET].values

    # preprocessing
    pipe0.fit(full_data[features],ytrain)
    # transform training data
    xtrain = pipe0.transform(xtrain)
    # transform validation data
    xvalid = pipe0.transform(xvalid)
    # initialize Random Forest model
    # if opt_model is not None:
        # model = opt_model
    # else:
        # model = RandomForestClassifier()
    model = XGBClassifier(**param)

    # fit model on training data
    model.fit(xtrain, ytrain)
    # predict on validation data
    # we need the probability values as we are calculating AUC
    # we will use the probability of 1s
    valid_preds = model.predict_proba(xvalid)[:, 1]
    # get roc auc score
    auc = metrics.roc_auc_score(yvalid, valid_preds)
    # print auc
    print(f'fold={fold}, auc={auc}')
    
    # # save the model
    # joblib.dump(pipe0,os.path.join(config.PIPE, f"pipe_{fold}.bin"))
    # joblib.dump(model,os.path.join(config.MODEL, f"{model_name}_{fold}.bin"))
    return auc

def objective(trial):

    param = {
        "verbosity": 0,
        "tree_method": 'exact',
        "use_label_encoder": False,
        "booster": "gbtree",
        "tree_method":'gpu_hist', 
        "gpu_id":0,
        "eval_metric":trial.suggest_categorical("eval_metric", ["error", "logloss","auc","aucpr"]),
        "max_depth" : trial.suggest_int("max_depth", 2, 4),
        "n_estimators" : trial.suggest_int("n_estimators", 10, 1000),
        "gamma" : trial.suggest_float("gamma", 1e-8, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0),        
        "learning_rate": trial.suggest_float("learning_rate", 1e-8, 1.0),
    }
    all_aucs = []
    for f in range(5):
        temp_auc = run_training(fold=f,model_name="rf",param=param)
        all_aucs.append(temp_auc)

    return np.mean(all_aucs)


def predict(model_name="mdoel"):
    df_test = pd.read_csv(config.TEST_DATA)
    ss = pd.read_csv(config.SAMPLE_SUB)
    # feature columns
    features = [
    f for f in df_test.columns if f not in ("ID")
    ]
    
    fold = 0
    pipe0 = joblib.load(os.path.join(config.PIPE, f"pipe_{fold}.bin"))
    model = joblib.load(os.path.join(config.MODEL, f"rf_{fold}.bin"))

    xtest = pipe0.transform(df_test[features])
    ytest = model.predict(xtest)

    ss["Is_Lead"] = ytest
    ss.to_csv(os.path.join(config.SUB,f"submission_{model_name}_{fold}.csv"),index=False)


if __name__ == '__main__':
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)

    print("best trials:")
    trial_ = study.best_trial

    print(trial_.values) 
    print(trial_.params) 
