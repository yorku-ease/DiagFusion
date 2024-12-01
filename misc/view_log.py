import pickle
import numpy as np

# with open("data/gaia/test/stratification_texts.pkl", "rb") as f:
#     data = pickle.load(f)
#     print(data)
data = np.load("data/gaia/test/parse/stratification_texts.pkl", allow_pickle=True)
print(data)
