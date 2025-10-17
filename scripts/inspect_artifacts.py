import joblib
import pandas as pd
import numpy as np
from pathlib import Path

MODEL_PATH = Path('src') / 'models' / 'LogisticRegression.pkl'
PREPROC_PATH = Path('src') / 'models' / 'preprocessor.pkl'

print('MODEL_PATH', MODEL_PATH.exists(), MODEL_PATH)
print('PREPROC_PATH', PREPROC_PATH.exists(), PREPROC_PATH)

model = None
pre = None

try:
    model = joblib.load(MODEL_PATH)
    print('Loaded model type:', type(model))
except Exception as e:
    print('Model load error:', e)

try:
    pre = joblib.load(PREPROC_PATH)
    print('Loaded preprocessor type:', type(pre))
except Exception as e:
    print('Preprocessor load error:', e)

# show attributes
if pre is not None:
    print('preprocessor has get_feature_names_out:', hasattr(pre, 'get_feature_names_out'))
    print('preprocessor has feature_names_in_:', hasattr(pre, 'feature_names_in_'))

if model is not None:
    print('model has feature_names_in_:', hasattr(model, 'feature_names_in_'))

# Build a tiny sample df similar to previous tests
sample = pd.DataFrame([{'gender':'Female','SeniorCitizen':0,'tenure':1,'MonthlyCharges':29.85,'TotalCharges':29.85}])
print('sample columns:', sample.columns.tolist())

if pre is not None:
    # Ensure sample has all input columns expected by the preprocessor
    try:
        input_cols = None
        if hasattr(pre, 'feature_names_in_'):
            input_cols = list(pre.feature_names_in_)
            print('preprocessor.feature_names_in_ length:', len(input_cols))
        if input_cols is not None:
            for c in input_cols:
                if c not in sample.columns:
                    sample[c] = pd.NA
            # Reorder columns to match expected
            sample = sample[input_cols]
            print('sample columns after ensure:', sample.columns.tolist()[:20], '... total', len(sample.columns))

        X = pre.transform(sample)
        print('transform output type:', type(X))
        # If sparse, convert to array for shape
        try:
            import scipy.sparse as sp
            if sp.issparse(X):
                print('transform returned sparse matrix; converting to array for inspection')
                X_arr = X.toarray()
            else:
                X_arr = X
        except Exception:
            X_arr = X
        try:
            print('transform output shape:', getattr(X_arr, 'shape', None))
        except Exception:
            pass
    except Exception as e:
        print('transform failed after ensuring columns:', e)

    try:
        if hasattr(pre, 'get_feature_names_out'):
            try:
                    fn = pre.get_feature_names_out(sample.columns)
                    print('get_feature_names_out(sample.columns) length:', len(fn))
                    print('first 20 feature names:', fn[:20])
            except Exception as e:
                print('get_feature_names_out(sample.columns) error:', e)
            try:
                fn2 = pre.get_feature_names_out()
                print('get_feature_names_out() length:', len(fn2))
                print('first 20 feature names:', fn2[:20])
            except Exception as e:
                print('get_feature_names_out() error:', e)
    except Exception as e:
        print('feature name inspection error:', e)

if model is not None:
    try:
        print('model.feature_names_in_:', getattr(model, 'feature_names_in_', None))
    except Exception as e:
        print('model.feature_names_in_ error:', e)

print('Done')
