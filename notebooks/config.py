import os


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_DATA = os.path.join(ROOT_DIR,'input','train_s3TEQDk.csv')
TRAIN_FOLDS = os.path.join(ROOT_DIR,'input','train_folds.csv')

TEST_DATA = os.path.join(ROOT_DIR,'input','test_mSzZ8RL.csv')


if __name__ == '__main__':
    print(ROOT_DIR)