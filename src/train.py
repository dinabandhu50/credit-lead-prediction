import os
import config
import joblib
import pandas as pd
from pipeline import pipe0
import model_dispatcher


from sklearn import metrics

def run_trainig(fold,model_name="model"):
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
    # initialize Logistic Regression model
    clf = model_dispatcher.models[model_name]

    # fit model on training data
    clf.fit(xtrain, ytrain)
    # predict on validation data
    # we need the probability values as we are calculating AUC
    # we will use the probability of 1s
    valid_preds = clf.predict_proba(xvalid)[:, 1]
    # get roc auc score
    auc = metrics.roc_auc_score(yvalid, valid_preds)
    # print auc
    print(f'fold={fold}, auc={auc}')
    
    # save the model
    joblib.dump(pipe0,os.path.join(config.PIPE, f"pipe0_{fold}.bin"))
    joblib.dump(clf,os.path.join(config.MODEL, f"{model_name}_{fold}.bin"))


def predict(model_name="model"):
    df_test = pd.read_csv(config.TEST_DATA)
    ss = pd.read_csv(config.SAMPLE_SUB)
    # feature colymns
    features = [
    f for f in df_test.columns if f not in ("ID")
    ]
    
    fold = 0
    pipe0 = joblib.load(os.path.join(config.PIPE, f"pipe0_{fold}.bin"))
    model = joblib.load(os.path.join(config.MODEL, f"{model_name}_{fold}.bin"))

    xtest = pipe0.transform(df_test[features])
    ytest = model.predict(xtest)

    ss["Is_Lead"] = ytest
    ss.to_csv(os.path.join(config.SUB,f"submission_{model_name}_{fold}.csv"),index=False)
 

if __name__ == '__main__':
    model_name = "rf"
    # model_name = "xgb"
    # model_name = "cat"

    for f in range(5):
        run_trainig(fold=f,model_name=model_name)
    predict(model_name=model_name)