from sklearn.model_selection import train_test_split
from umap import UMAP
import numpy as np
import pickle
from joblib import Parallel, delayed

folder = './graphs/basic_test/'
with open(folder + 'combined_data_basic.pickle', 'rb') as fil:
    basic_UNsorted = pickle.load(fil)

basic_UNsorted = np.array(basic_UNsorted)
basic_UNsorted = basic_UNsorted[~np.isnan(basic_UNsorted).any(axis=1)]

# Split the data
X_train, X_val = train_test_split(basic_UNsorted, test_size=0.2)


def evaluate_umap(n_neighbors, min_dist, n_components, X_train, X_val):
    model = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, 
                 force_approximation_algorithm=True)
    model.fit(X_train)
    score = model.transform(X_val).std()
    print(score, n_neighbors, min_dist, n_components)
    return (score, n_neighbors, min_dist, n_components)

# Hyperparameters to test
n_neighbors_options = [5, 6, 7]
min_dist_options = [0.1, 0.9, 10.0]
n_components_options = [3, 4, 5, 6, 7]  # Example dimensions

# Parallelize the search
results = Parallel(n_jobs=-1)(delayed(evaluate_umap)(n_neighbors, min_dist, n_components, X_train, X_val)
                               for n_neighbors in n_neighbors_options
                               for min_dist in min_dist_options
                               for n_components in n_components_options)
# Find the best parameters
# best_score, best_params = min((res[0], res[1:]) for res in results)

# Fit the model with best parameters
# model = UMAP(n_neighbors=best_params[0], min_dist=best_params[1], n_components=best_params[2])
# model = UMAP(n_neighbors=5, min_dist=1.0, n_components=6)
# out = model.fit(basic_UNsorted)

# print(out)
# print(np.shape(out.embedding_))

# # Save the model
# with open('umap_model_basic.pickle', 'wb') as f:
#     pickle.dump(model, f)