from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

@api_view(['GET'])
def hello(request):
    return Response({'msg': 'Hello, world!'})

@api_view(['POST'])
def hello(request):
    
    return Response({'msg': 'File at backend'})
    #write code for five number summary


#knn - use selected dataset
@api_view(['POST'])
def knn(request):
    dataset = request.data['dataset']
    # k = request.data['k']
    k=3
    # return Response({'mean' : request.data['dataset']})

    if dataset == "Iris Dataset":
        data = datasets.load_iris()
        X = data.data
        y = data.target
    elif dataset == "Breast Cancer Dataset":
        data = datasets.load_breast_cancer()
        X = data.data
        y = data.target
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_pred = []
    for x in X_test:
        # Compute distances between x and all examples in the training set
        distances = np.linalg.norm(X_train - x, axis=1)
        # Sort by distance and return indices of the first k neighbors
        k_indices = np.argsort(distances)[:k]
        # Extract the labels of the k nearest neighbor training samples
        k_nearest_labels = [y_train[i] for i in k_indices]
        # print("k nearest labels ", k_nearest_labels)
        # Return the most common class label
        most_common = np.bincount(k_nearest_labels).argmax()
        y_pred.append(most_common)
    
    acc = accuracy_score(y_test, y_pred)
    print(acc)
    return Response({'acc' : acc})

    # Calculate mean and standard deviation for each class and feature
    # class_means = []
    # class_stds = []

    # for i in np.unique(y_train):
    #     class_means.append(np.mean(X_train[y_train == i], axis=0))
    #     class_stds.append(np.std(X_train[y_train == i], axis=0))

    # class_means = np.array(class_means)
    # class_stds = np.array(class_stds)

    # # Function to calculate Gaussian probability
    # def calculate_probability(x, mean, std):
    #     exponent = np.exp(-((x - mean) ** 2) / (2 * (std ** 2)))
    #     return (1 / (np.sqrt(2 * np.pi) * std)) * exponent

    # # Function to predict the class for a given sample
    # def predict(sample):
    #     probabilities = []

    #     for i in range(len(np.unique(y_train))):
    #         # Calculate class probabilities using Bayes' theorem
    #         prior = len(X_train[y_train == i]) / len(X_train)
    #         likelihood = np.prod(calculate_probability(sample, class_means[i], class_stds[i]))
    #         posterior = prior * likelihood
    #         probabilities.append(posterior)

    #     # Return the class with the highest probability
    #     return np.argmax(probabilities)

    # # Make predictions on the test set
    # predictions = [predict(sample) for sample in X_test]

    # # Calculate accuracy
    # accuracy = np.sum(predictions == y_test) / len(y_test)
    # print(f"Accuracy: {accuracy * 100:.2f}%")
    # return Response({'acc' : accuracy})  








#decision tree 

# import numpy as np
# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split

# # Load Iris dataset
# iris = load_iris()
# X = iris.data
# y = iris.target

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Decision Tree Node class
# class Node:
#     def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
#         self.feature_index = feature_index  # Index of feature to split on
#         self.threshold = threshold  # Threshold value for the feature
#         self.left = left  # Left subtree
#         self.right = right  # Right subtree
#         self.value = value  # Class label for leaf nodes

# # Decision Tree class
# class DecisionTree:
#     def __init__(self, max_depth=None):
#         self.max_depth = max_depth  # Maximum depth of the tree
#         self.root = None  # Root node of the tree

#     def fit(self, X, y):
#         self.root = self._build_tree(X, y, depth=0)

#     def _build_tree(self, X, y, depth):
#         # Stopping criteria
#         if depth == self.max_depth or len(np.unique(y)) == 1:
#             return Node(value=np.argmax(np.bincount(y)))

#         # Find the best split
#         feature_index, threshold = self._find_best_split(X, y)

#         # Split the data
#         X_left, y_left, X_right, y_right = self._split_data(X, y, feature_index, threshold)

#         # Recursively build the left and right subtrees
#         left_subtree = self._build_tree(X_left, y_left, depth + 1)
#         right_subtree = self._build_tree(X_right, y_right, depth + 1)

#         # Create a new node for the decision tree
#         return Node(feature_index=feature_index, threshold=threshold, left=left_subtree, right=right_subtree)

#     def _find_best_split(self, X, y):
#         num_features = X.shape[1]
#         best_feature_index = None
#         best_threshold = None
#         best_gini = float('inf')

#         for feature_index in range(num_features):
#             thresholds = np.unique(X[:, feature_index])
#             for threshold in thresholds:
#                 X_left, y_left, X_right, y_right = self._split_data(X, y, feature_index, threshold)
#                 gini = self._calculate_gini(y_left, y_right)
#                 if gini < best_gini:
#                     best_gini = gini
#                     best_feature_index = feature_index
#                     best_threshold = threshold

#         return best_feature_index, best_threshold

#     def _split_data(self, X, y, feature_index, threshold):
#         left_mask = X[:, feature_index] <= threshold
#         right_mask = ~left_mask
#         return X[left_mask], y[left_mask], X[right_mask], y[right_mask]

#     def _calculate_gini(self, y_left, y_right):
#         size_left = len(y_left)
#         size_right = len(y_right)
#         total_size = size_left + size_right

#         gini_left = 1.0 - sum((np.sum(y_left == c) / size_left) ** 2 for c in np.unique(y_left))
#         gini_right = 1.0 - sum((np.sum(y_right == c) / size_right) ** 2 for c in np.unique(y_right))

#         gini = (size_left / total_size) * gini_left + (size_right / total_size) * gini_right
#         return gini

#     def predict(self, X):
#         return np.array([self._predict(x, self.root) for x in X])

#     def _predict(self, x, node):
#         if node.value is not None:
#             return node.value

#         if x[node.feature_index] <= node.threshold:
#             return self._predict(x, node.left)
#         else:
#             return self._predict(x, node.right)

# # Create and train the decision tree
# tree = DecisionTree(max_depth=3)
# tree.fit(X_train, y_train)

# # Make predictions on the test set
# predictions = tree.predict(X_test)

# # Calculate accuracy
# accuracy = np.sum(predictions == y_test) / len(y_test)
# print(f"Accuracy: {accuracy * 100:.2f}%")













#birch

# import numpy as np
# from sklearn.datasets import make_blobs

# class BIRCHNode:
#     def __init__(self, threshold, branching_factor):
#         self.threshold = threshold
#         self.branching_factor = branching_factor
#         self.children = []
#         self.num_points = 0
#         self.subcluster_centers = None
#         self.subcluster_labels = None

#     def insert(self, point):
#         if not self.children:
#             if self.subcluster_centers is None:
#                 self.subcluster_centers = point
#             else:
#                 self.subcluster_centers = np.vstack([self.subcluster_centers, point])

#             self.num_points += 1

#             if self.num_points > self.threshold:
#                 self.split()
#         else:
#             closest_child = self.find_closest_child(point)
#             closest_child.insert(point)

#     def split(self):
#         children_centers = self.subcluster_centers
#         children_labels = np.arange(self.num_points)

#         self.subcluster_centers = None
#         self.subcluster_labels = None
#         self.num_points = 0

#         for i in range(0, len(children_centers), self.branching_factor):
#             child = BIRCHNode(self.threshold, self.branching_factor)
#             child.subcluster_centers = children_centers[i:i + self.branching_factor]
#             child.subcluster_labels = children_labels[i:i + self.branching_factor]
#             child.num_points = len(child.subcluster_labels)
#             self.children.append(child)

#     def find_closest_child(self, point):
#         distances = [np.linalg.norm(point - child.subcluster_centers.mean(axis=0)) for child in self.children]
#         return self.children[np.argmin(distances)]


# class BIRCH:
#     def __init__(self, threshold, branching_factor):
#         self.root = BIRCHNode(threshold, branching_factor)

#     def fit(self, X):
#         for point in X:
#             self.root.insert(point)

#     def predict(self, X):
#         labels = []
#         for point in X:
#             current_node = self.root
#             while current_node.children:
#                 current_node = current_node.find_closest_child(point)
#             labels.append(current_node.subcluster_labels)

#         return np.concatenate(labels)


# # Example usage:
# # Generate sample data
# X, _ = make_blobs(n_samples=1000, centers=3, random_state=42)

# # Instantiate and fit the BIRCH algorithm
# birch = BIRCH(threshold=5, branching_factor=10)
# birch.fit(X)

# # Predict cluster labels
# cluster_labels = birch.predict(X)

# # Print the cluster labels
# print("Cluster labels:", cluster_labels)
