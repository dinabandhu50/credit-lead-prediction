import os

# paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_DATA = os.path.join(ROOT_DIR,'input','train_s3TEQDk.csv')
TRAIN_FOLDS = os.path.join(ROOT_DIR,'input','train_folds.csv')

TEST_DATA = os.path.join(ROOT_DIR,'input','test_mSzZ8RL.csv')

SAMPLE_SUB = os.path.join(ROOT_DIR,'input','sample_submission_eyYijxG.csv')
SUB = os.path.join(ROOT_DIR,'output')

MODEL = os.path.join(ROOT_DIR,'models')
PIPE = os.path.join(ROOT_DIR,'models')

# columns
CAT_COLS = ["Gender","Region_Code","Occupation","Channel_Code","Credit_Product","Is_Active"]
NUM_COLS = ["Age","Vintage","Avg_Account_Balance"]
TARGET = "Is_Lead"

# missing col
MISSING_COLS = ["Credit_Product"]
OHE_COLS = ["Gender","Occupation","Channel_Code","Credit_Product","Is_Active"]
COUNT_FREQ_COLS = ["Region_Code"]
YJT_COLS = ["Avg_Account_Balance"]


if __name__ == '__main__':
    print(ROOT_DIR)