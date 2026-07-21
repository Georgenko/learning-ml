import numpy as np

MIN_SAMPLES_SPLIT = 10
MAX_DEPTH = 3
MAX_FEATURES = 3

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.is_leaf = self.value is not None
        # A leaf is created by calling Node(value=X)
        # A node is created by calling Node(feature=X, threshold=X, left=X, right=X)

    def predict_one(self, x):
        if self.is_leaf:
            return self.value
        else:
            if x[self.feature] <= self.threshold:
                return self.left.predict_one(x)
            else:
                return self.right.predict_one(x)

    def predict_batch(self, X):
        return np.array([self.predict_one(X[i,:]) for i in range(X.shape[0])])

    def print_tree_sklearn_style(self, feature_names, depth=0):
        indent = "|   " * depth
        if self.is_leaf:
            print(f"{indent}|--- class: {self.value}")
        else:
            fname = feature_names[self.feature]
            print(f"{indent}|--- {fname} <= {self.threshold:.2f}")
            self.left.print_tree_sklearn_style(feature_names, depth + 1)
            print(f"{indent}|--- {fname} >  {self.threshold:.2f}")
            self.right.print_tree_sklearn_style(feature_names, depth + 1)

def build_tree(X, y, rng=None, max_features=None, depth=0, max_depth=None, min_samples_split=MIN_SAMPLES_SPLIT):
    if min_samples_split < 2:
        raise ValueError("min_samples_split cannot be lower than 2!")

    # stopping conditions
    reached_max_depth = (depth == max_depth) if max_depth else False
    too_little_samples_left = len(y) < min_samples_split
    all_samples_same_class = len(np.unique(y)) == 1
    if reached_max_depth or too_little_samples_left or all_samples_same_class:
        return Node(value=most_common_label(y)) # leaf

    lowest_weighted_gini, feature, thresh = find_best_split(X, y, rng, max_features)

    # another stopping condition
    no_improvement_in_splitting = lowest_weighted_gini >= gini(y)
    if no_improvement_in_splitting:
        return Node(value=most_common_label(y)) # leaf

    # split X and y
    mask = X[:, feature] <= thresh
    X_left = X[mask, :]
    y_left = y[mask]
    X_right = X[~mask, :]
    y_right = y[~mask]

    # recurse
    left_node = build_tree(X_left, y_left, rng, max_features, depth + 1, max_depth, min_samples_split)
    right_node = build_tree(X_right, y_right, rng, max_features, depth + 1, max_depth, min_samples_split)

    return Node(feature=feature, threshold=thresh, left=left_node, right=right_node)

def most_common_label(labels):
    unique_vals, counts = np.unique(labels, return_counts=True)
    return unique_vals[np.argmax(counts)]

def find_best_split(X, y, rng=None, max_features=None):
    lowest_gini = np.inf
    feature_at_lowest_gini = None
    thresh_at_lowest_gini = None

    n_features = X.shape[1]

    if max_features is None:
        candidate_features = range(n_features)
    else:
        if rng is None:
            raise ValueError("rng must be provided when max_features is set")
        if max_features > n_features:
            raise ValueError("max_features cannot be bigger than n_features!")
        candidate_features = rng.choice(n_features, size=max_features, replace=False)

    for feature_idx in candidate_features:

        unique_sorted_vals = np.unique(X[:, feature_idx])
        thresholds = (unique_sorted_vals[:-1] + unique_sorted_vals[1:]) / 2

        for threshold in thresholds:
            y_left, y_right = split_labels(y, X[:,feature_idx], threshold)
            current_gini = weighted_gini(y_left, y_right)

            if current_gini < lowest_gini:
                lowest_gini = current_gini
                feature_at_lowest_gini = feature_idx
                thresh_at_lowest_gini = threshold
    return lowest_gini, feature_at_lowest_gini, thresh_at_lowest_gini

def split_labels(y, feature, threshold):
    y_left = y[feature <= threshold]
    y_right = y[feature > threshold]
    return y_left, y_right

def weighted_gini(y_left, y_right):
    len_left = len(y_left)
    len_right = len(y_right)
    len_total = len_left + len_right
    return (len_left * gini(y_left) + len_right * gini(y_right)) / len_total

"""Gini impurity is,
if you take a random sample from a dataset,
and assign it randomly to a class,
what is the probability that this class is wrong?"""
def gini(labels):
    labels_count = len(labels)
    if labels_count == 0:
        return 0
    _, counts = np.unique(labels, return_counts=True)
    probabilities = np.sum((counts / labels_count) ** 2)
    return 1 - probabilities

