# %%
from helper.config import DEBUG, DATA_PATH
from helper.reader import PSBDataset
dataset = PSBDataset(DATA_PATH)
dataset.read()
# dataset.load_files_in_memory()

# %%
import pickle
data_subset = [dict(**item["meta_data"]) for item in dataset.full_data]
pickle.dump(data_subset, open("dataset.pkl", "wb"))