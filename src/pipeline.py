import numpy as np
import config
from sklearn.pipeline import Pipeline

from feature_engine.imputation import CategoricalImputer
from feature_engine.encoding import OneHotEncoder
from feature_engine.encoding import CountFrequencyEncoder

from feature_engine.transformation import YeoJohnsonTransformer
from feature_engine.discretisation import ArbitraryDiscretiser


user_dict = {'Age': [0, 38, np.Inf]}
pipe0 = Pipeline([
    ('missing',CategoricalImputer(variables=config.MISSING_COLS,imputation_method='missing',fill_value='Missing')),
    ('ohe',OneHotEncoder(variables=config.OHE_COLS)),
    ('fe',CountFrequencyEncoder(encoding_method='frequency',variables=config.COUNT_FREQ_COLS)),
    ('yjt',YeoJohnsonTransformer(variables=config.YJT_COLS)),
    ('bin',ArbitraryDiscretiser(binning_dict=user_dict, return_object=False, return_boundaries=False)),
])
