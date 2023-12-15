from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
import json
from .models import CSVFile
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
from django.http import HttpResponse,JsonResponse, FileResponse
import scipy.stats
from rest_framework import status
import pandas as pd
import tensorflow as tf
import math
import os
import numpy as np
import statistics as st

from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text
from sklearn.tree import export_graphviz

from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

import graphviz
from sklearn import tree
import pydotplus
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, classification_report, mean_squared_error, mean_absolute_error, r2_score
from .models import AttributesOfDataset


class KNNClassifier():
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        # Compute distances between x and all examples in the training set
        distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]
        # np.sqrt(np.sum((x1 - x2) ** 2))
        
        # Sort by distance and return indices of the first k neighbors
        k_indices = np.argsort(distances)[:self.k]
        # Extract the labels of the k nearest neighbor training samples
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        print("k nearest labels ", k_nearest_labels)
        # Return the most common class label
        most_common = np.bincount(k_nearest_labels).argmax()
        print("most common ", most_common)
        return most_common
    
def calculate_mean(df):
    summ = sum(df)
    n = len(df)
    mean = round(summ/n, 2)
    return mean

# Calculate the median  
def calculate_median(df):
    sorted_data = sorted(df)
    # print(sorted_data)
    n = len(sorted_data)
    if n % 2 == 1:  # Odd number of elements
        median = sorted_data[n // 2]
    else:  # Even number of elements
        middle1 = sorted_data[n // 2 - 1]
        middle2 = sorted_data[n // 2]
        median = (middle1 + middle2) / 2
    return median

#calculate mode
def calculate_mode(df):
    freq={}
    for a in df:
        if a in freq:
            freq[a] += 1
        else:
            freq[a] = 1
    maxx = 0
    mode = 0.0    
    for a in freq:
        if(freq[a] > maxx):
            maxx = freq[a]
            mode = a
    
    return mode


#calculate mid range
def calculate_midrange(df):
    midrange = (min(df) + max(df))/2
    return midrange

#calculate variance
def calculate_variance(df):
    mean = calculate_mean(df)
    n = len(df)
    summ = 0
    for a in df:
        summ += (a - mean)**2
    variance = summ/n
    return variance

#calculate standard deviation
def calculate_sd(df):
    sd = math.sqrt(calculate_variance(df))
    return sd

#calculate range
def calculate_range(df):
    range = (max(df) - min(df))
    return range
 
# Calculate interquartile range  
def calculate_interquartile(df):
    sorted_data = sorted(df)
    print("sorted data for selected attribute ", sorted_data)
    n = len(sorted_data)
    q1 = sorted_data[n // 4]
    q3 = sorted_data[(n*3) // 4]
    iq = round((q3-q1), 2)
    return iq

#five-number summary
def calculate_fiveSumm(df):
    n = len(df)
    sorted_data = sorted(df)
    five = {
        'high': max(df),
        'q3': sorted_data[(n*3)//4],
        'median': calculate_median(df),
        'q1': sorted_data[n//4],
        'low': min(df)
    }
    return five

#second assignment q3
def min_max_normalization(data):
    min_val = min(data)
    max_val = max(data)
    normalized_data = [(x - min_val) / (max_val - min_val) for x in data]
    return normalized_data

def z_score_normalization(data):
    mean = np.mean(data)
    std_dev = np.std(data)
    normalized_data = [(x - mean) / std_dev for x in data]
    return normalized_data

def decimal_scaling_normalization(data):
    max_abs = max(abs(x) for x in data)
    d = len(str(int(max_abs)))
    normalized_data = [x / (10 ** d) for x in data]
    return normalized_data


#first assignment calculation
@api_view(['POST'])
def stats(request):
    df = json.loads(request.body)
    print(type(df[0]))
    print("sent data", df)
    mean = calculate_mean(df)
    median = calculate_median(df)
    mode = calculate_mode(df)
    midrange = calculate_midrange(df)
    variance = round(calculate_variance(df), 2)
    std = round(math.sqrt(calculate_variance(df)), 2)
    range = round(calculate_range(df), 2)
    interquartile = calculate_interquartile(df)
    fiveSumm = calculate_fiveSumm(df)
    minmaxnorm = min_max_normalization(df)
    zscorenorm = z_score_normalization(df)
    decscalingnorm = decimal_scaling_normalization(df)
    print("mean =", st.mean(df))
    print("median =", st.median(df))
    print("mode =", st.mode(df))
    print("var =", st.variance(df))
    print("std =", st.stdev(df))
    q1 = np.percentile(df, 25)
    q3 = np.percentile(df, 75)
    iqr = q3 - q1
    print("q1 =", q1)
    print("q3 =", q3)
    print("iqr =", iqr)
    
    myAtt = AttributesOfDataset.objects.get(id=1)
    return Response({
    'mean': mean,
    'median': median,
    'mode': mode,
    'midrange': midrange,
    'variance': variance,
    'std': std,
    'range': range,
    'interquartile': interquartile,
    'fiveSumm': fiveSumm,
    'attributes': myAtt.attributeList,
    #ass 2
    'df': sorted(df),
    'min_max_norm': sorted(minmaxnorm),
    'z_score_norm':sorted(zscorenorm),
    'dec_scaling_norm':sorted(decscalingnorm),
    }, status=status.HTTP_200_OK)

@api_view(['POST'])
def upload_fileIris(request):
    file = request.FILES['file']
    dfd = pd.read_csv(file)
    attributes = []
    for a in dfd:
        if a != 'Id':
            attributes.append(a)
    json_attributes = json.dumps(attributes)
    att = AttributesOfDataset(attributeList = json_attributes)
    att.save()
    return Response({'data':dfd, 'attributes':attributes}, status=status.HTTP_200_OK)

#2nd assignment
@api_view(['POST'])
def upload_fileBreast(request):
    file = request.FILES['file']
    dfd = pd.read_csv(file, delimiter=',')
    
    custom_column_names = ['class', 'age', 'menopause', 'tumor-size', 'inv-nodes', 'node-caps', 'deg-malig', 'breast', 'breast-quad', 'irradiat']
    dfd.columns = custom_column_names
    attributes = []
    for a in dfd:
        if a != 'Id':
            attributes.append(a)
    return Response({'data':dfd, 'attributes':attributes}, status=status.HTTP_200_OK)


def calculateCorr(twoattris):
    correlation_coefficient, p_value = scipy.stats.pearsonr(twoattris['att1'], twoattris['att2'])
    return {correlation_coefficient, p_value}

def chiSquare(twoattris):
    contingency_table = pd.crosstab(twoattris['att1'], twoattris['att2'])
    print("contigency table ")
    print(contingency_table)
    chi2, p, _, _ = scipy.stats.chi2_contingency(contingency_table)
    print(f'Chi-squared value: {chi2}')
    print(f'p-value: {p}')
    return (chi2, p)


@api_view(['POST'])
def stats2(request):
    twoattris = json.loads(request.body)
    if((type(twoattris['att1'][1]) == float or type(twoattris['att1'][1]) == int) and (type(twoattris['att2'][1]) == float or type(twoattris['att2'][1]) == int)):
        corr, p = calculateCorr(twoattris=twoattris)
        return Response({
            'coef': corr,
            'p':p
        }, status=status.HTTP_200_OK)
    chi, pc = 0, 0
    chi, pc = chiSquare(twoattris=twoattris)
    return Response({
        'chi': chi,
        'pc' : pc,
    }, status=status.HTTP_200_OK)

@api_view(['POST'])
def normalise(request):
    dfd = json.loads(request.body)
    ds = dfd['dataset']
    attname1 = dfd['xaxisSc']
    attname2 = dfd['yaxisSc']
    norm1 = min_max_normalization(ds[attname1])
    norm2 = min_max_normalization(ds[attname2])
    return Response({'norm1': norm1, 'norm2':norm2}, status=status.HTTP_200_OK)
   
@api_view(['POST']) 
def stats3(request):
    twoattris = json.loads(request.body)
    attribute_selection_measure = twoattris['att2']
    if (twoattris['att1'] == 'Breast Cancer'):
        return breastCancer(attribute_selection_measure)
    elif (twoattris['att1'] == 'Car Evaluation'):
        return carEvaluation(attribute_selection_measure)
    elif (twoattris['att1'] == 'Balance Scale'):
        return balanceScale(attribute_selection_measure) 
    

def balanceScale(attribute_selection_measure):
    # Load the dataset (you can replace this with your dataset)
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data"
    column_names = ["class", "Left-Weight", "Left-Distance", "Right-Weight", "Right-Distance"]
    df = pd.read_csv(url, names=column_names)
    # Split the dataset into features (X) and target (y)
    X = df.drop('class', axis=1)
    y = df['class']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a decision tree classifier
    # Create a DecisionTreeClassifier with the chosen attribute selection measure
    if attribute_selection_measure == 'Information gain':
        clf = DecisionTreeClassifier(criterion='entropy')
    elif attribute_selection_measure == 'Gain ratio':
        clf = DecisionTreeClassifier(criterion='entropy', splitter='best')
    elif attribute_selection_measure == 'Gini index':
        clf = DecisionTreeClassifier(criterion='gini')

    # Fit the classifier to the training data
    clf.fit(X_train, y_train)

    #tree
    decision_tree_text = export_text(clf, feature_names=X.columns.tolist())
    tree.plot_tree(clf, filled=True, feature_names=X.columns.tolist(), class_names=list(map(str, clf.classes_)))
    
    # Create a Graphviz graph
    dot_data = tree.export_graphviz(clf, out_file=None, filled=True, feature_names=X.columns.tolist(), class_names=list(map(str, clf.classes_)))
    graph = pydotplus.graph_from_dot_data(dot_data)
    
    tree_image_path = 'static/plot/image.png'
    graph.write_png(tree_image_path)

    # Make predictions on the test data
    y_pred = clf.predict(X_test)

    # Evaluate the classifier
    class_report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred) # tp + tn/p+n
    misclassification_rate = 1 - accuracy  # fp + fn/p+n
    sensitivity = recall_score(y_test, y_pred,average='macro')   # tp / tp + fn  true positivity scor
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])  # tn / tn + fp  true negativity score
    precision = precision_score(y_test, y_pred, average='macro')  # tp / tp + fp Positive Predictive Value
    recall = sensitivity  # Recall is the same as sensitivity
    
    #extract rules
    rules = []
    rules = extract_rules(tree=clf.tree_, feature_names=X.columns, class_names=clf.classes_.astype(str))
    print("ruulless", rules)
    predictions = []
    # Loop through each row in your one-hot encoded dataset
    for index, row in X_test.iterrows():
        # Convert the row to a dictionary
        data_point = row.to_dict()
        
        # Pass data_point to your rule-based classifier function and append the result to the predictions list
        prediction = rule_based_classifier(rules, data_point)
        predictions.append(prediction)

    print(predictions)
# Now, the predictions list contains the predictions for each data point
    accuracyRule = accuracy_score(predictions, y_test)
    print("Accuracy of Rule Based Classifier :", accuracyRule)
    print("Accuracy of balance scale:", accuracy)
    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:\n", class_report)
    
    coverage = calculate_coverage(rules, df)
    print("coverage ", coverage)
    toughness = calculate_toughness(rules, df, y)
    print("toughness ", toughness)
    return Response({
        'tree': decision_tree_text,
        'cm' : cm,
        'acc' : accuracy,
        'misclass': misclassification_rate,
        'sens' : sensitivity,
        'spec' : specificity,
        'recall' : recall,
        'prec' : precision,
        'rules': rules,
        'accuracyRule' : accuracyRule,
        'coverage' : coverage,
        'toughness': toughness
        
    }, status=status.HTTP_200_OK)
    

def breastCancer(attribute_selection_measure):
    # Load the dataset (you can replace this with your dataset)
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer/breast-cancer.data"
    column_names = ['class', 'age', 'menopause', 'tumor-size', 'inv-nodes', 'node-caps', 'deg-malig', 'breast', 'breast-quad', 'irradiat']
    df = pd.read_csv(url, names=column_names)
    df_encoded = pd.get_dummies(df, columns=[ 'age', 'menopause', 'tumor-size', 'inv-nodes', 'node-caps', 'deg-malig', 'breast', 'breast-quad', 'irradiat'])

    # Split the dataset
    X = df_encoded.drop('class', axis=1)
    y = df_encoded['class']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a decision tree classifier
    if attribute_selection_measure == 'Information gain':
        clf = DecisionTreeClassifier(criterion='entropy')
    elif attribute_selection_measure == 'Gain ratio':
        clf = DecisionTreeClassifier(criterion='entropy', splitter='best')
    elif attribute_selection_measure == 'Gini index':
        clf = DecisionTreeClassifier(criterion='gini')

    # Fit the classifier to the training data
    clf.fit(X_train, y_train)

    decision_tree_text = export_text(clf, feature_names=X.columns.tolist())
    tree.plot_tree(clf, filled=True, feature_names=X.columns.tolist(), class_names=list(map(str, clf.classes_)))
    
    # Create a Graphviz graph
    dot_data = tree.export_graphviz(clf, out_file=None, filled=True, feature_names=X.columns.tolist(), class_names=list(map(str, clf.classes_)))
    graph = pydotplus.graph_from_dot_data(dot_data)
    
    tree_image_path = 'static/plot/image.png'
    graph.write_png(tree_image_path)

    # Make predictions on the test data
    y_pred = clf.predict(X_test)

    # Evaluate the classifier
    class_report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred) # tp + tn/p+n
    misclassification_rate = 1 - accuracy  # fp + fn/p+n
    sensitivity = recall_score(y_test, y_pred,average='macro')   # tp / tp + fn  true positivity scor
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])  # tn / tn + fp  true negativity score
    precision = precision_score(y_test, y_pred, average='macro')  # tp / tp + fp Positive Predictive Value
    recall = sensitivity  # Recall is the same as sensitivity

     #extract rules
    rules = []
    rules = extract_rules(tree=clf.tree_, feature_names=X.columns, class_names=clf.classes_.astype(str))
    print("ruulless", rules)
    predictions = []
    # Loop through each row in your one-hot encoded dataset
    for index, row in X_test.iterrows():
        # Convert the row to a dictionary
        data_point = row.to_dict()
        
        # Pass data_point to your rule-based classifier function and append the result to the predictions list
        prediction = rule_based_classifier(rules, data_point)
        predictions.append(prediction)

    print(predictions)
# Now, the predictions list contains the predictions for each data point
    accuracyRule = accuracy_score(predictions, y_test)
    print("Accuracy of Rule Based Classifier :", accuracyRule)
    print("Accuracy of breast Cancer:", accuracy)
    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:\n", class_report)
    
    coverage = calculate_coverage(rules, df_encoded)
    print("coverage ", coverage)
    toughness = calculate_toughness(rules, df_encoded, y)
    print("toughness ", toughness)
    return Response({
        'tree': decision_tree_text,
        'cm' : cm,
        'acc' : accuracy,
        'misclass': misclassification_rate,
        'sens' : sensitivity,
        'spec' : specificity,
        'recall' : recall,
        'prec' : precision,
        'rules': rules,
        'accuracyRule' : accuracyRule,
        'coverage' : coverage,
        'toughness': toughness
        
    }, status=status.HTTP_200_OK)

def carEvaluation(attribute_selection_measure):
    # Load the dataset (you can replace this with your dataset)
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
    column_names =["buying", "maint", "doors","persons","lug_boot","safety","class"]
    df = pd.read_csv(url, names=column_names)
    # Perform one-hot encoding on the categorical features
    df_encoded = pd.get_dummies(df, columns=["buying", "maint", "doors", "persons", "lug_boot", "safety"])
    # Split the dataset
    X = df_encoded.drop("class", axis=1)
    y = df_encoded["class"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a decision tree classifier
    if attribute_selection_measure == 'Information gain':
        clf = DecisionTreeClassifier(criterion='entropy')
    elif attribute_selection_measure == 'Gain ratio':
        clf = DecisionTreeClassifier(criterion='entropy', splitter='best')
    elif attribute_selection_measure == 'Gini index':
        clf = DecisionTreeClassifier(criterion='gini')

    # Fit the classifier to the training data
    clf.fit(X_train, y_train)

    decision_tree_text = export_text(clf, feature_names=X.columns.tolist())
    tree.plot_tree(clf, filled=True, feature_names=X.columns.tolist(), class_names=list(map(str, clf.classes_)))
    dot_data = tree.export_graphviz(clf, out_file=None, filled=True, feature_names=X.columns.tolist(), class_names=list(map(str, clf.classes_)))
    graph = pydotplus.graph_from_dot_data(dot_data)
    
    tree_image_path = 'static/plot/image.png'
    graph.write_png(tree_image_path)

    # Make predictions on the test data
    y_pred = clf.predict(X_test)

    # Evaluate the classifier
    class_report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred) # tp + tn/p+n
    misclassification_rate = 1 - accuracy  # fp + fn/p+n
    sensitivity = recall_score(y_test, y_pred,average='macro')   # tp / tp + fn  true positivity scor
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])  # tn / tn + fp  true negativity score
    precision = precision_score(y_test, y_pred, average='macro')  # tp / tp + fp Positive Predictive Value
    recall = sensitivity  # Recall is the same as sensitivity

    print("Accuracy of car Evaluation:", accuracy)
    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:\n", class_report)
    
    #extract rules
    rules = []
    rules = extract_rules(tree=clf.tree_, feature_names=X.columns, class_names=clf.classes_.astype(str))
    print("ruulless", rules)
    predictions = []
    # Loop through each row in your one-hot encoded dataset
    for index, row in X_test.iterrows():
        # Convert the row to a dictionary
        data_point = row.to_dict()
        
        # Pass data_point to your rule-based classifier function and append the result to the predictions list
        prediction = rule_based_classifier(rules, data_point)
        predictions.append(prediction)

    # print(predictions)
    
# Now, the predictions list contains the predictions for each data point
    accuracyRule = accuracy_score(predictions, y_test)
    print("Accuracy of Rule Based Classifier :", accuracyRule)
    coverage = calculate_coverage(rules, df_encoded)
    print("coverage ", coverage)
    toughness = calculate_toughness(rules, df_encoded, y)
    print("toughness ", toughness)
    return Response({
        'tree': decision_tree_text,
        'cm' : cm,
        'acc' : accuracy,
        'misclass': misclassification_rate,
        'sens' : sensitivity,
        'spec' : specificity,
        'recall' : recall,
        'prec' : precision,
        'rules': rules,
        'accuracyRule' : accuracyRule,
        'coverage' : coverage,
        'toughness': toughness
        
    }, status=status.HTTP_200_OK)
 
    
#assignment 4
def extract_rules(tree, feature_names, class_names):
    rules = []

    def recurse(node, rule):
        if tree.children_left[node] == tree.children_right[node]:  # Leaf node
            class_idx = tree.value[node].argmax()
            class_name = class_names[class_idx]
            rules.append((rule, class_name))
        else:
            feature = feature_names[tree.feature[node]]
            threshold = tree.threshold[node]
            left_rule = f"{feature} <= {threshold}"
            right_rule = f"{feature} > {threshold}"
            recurse(tree.children_left[node], rule + [left_rule])
            recurse(tree.children_right[node], rule + [right_rule])

    recurse(0, [])
    return rules



def rule_based_classifier(rules, data_point):
    # Initialize a default class (e.g., "unknown" or "no decision")
    default_class = "unknown"
    
    # Iterate through the rules
    for rule_conditions, rule_class in rules:
        # Initialize a flag to check if all conditions in the rule are met
        conditions_met = True
        
        # Evaluate each condition in the rule
        for condition in rule_conditions:
            feature, comparison, threshold = condition.split()
            feature_value = data_point.get(feature, None)
            
            if feature_value is None:
                # The feature is not present in the data point; skip this rule
                conditions_met = False
                break
            
            if comparison == "<=":
                if feature_value > float(threshold):
                    # Condition not met
                    conditions_met = False
                    break
            elif comparison == ">":
                if feature_value <= float(threshold):
                    # Condition not met
                    conditions_met = False
                    break
        
        # If all conditions in the rule are met, return the corresponding class
        if conditions_met:
            return rule_class
    
    # If no rule matched, return the default class
    return default_class

def calculate_coverage(rules, dataset):
    covered_count = 0
    for index, row in dataset.iterrows():
        # Convert the row to a dictionary
        data_point = row.to_dict()
        if any(rule_based_classifier(rules, data_point)):
            covered_count += 1
    coverage = covered_count / len(dataset)
    return coverage

def calculate_toughness(rules, dataset, ground_truth_labels):
    correctly_classified_count = 0
    covered_count = 0
    for i, row in dataset.iterrows():
        # Convert the row to a dictionary
        data_point = row.to_dict()
        predicted_label = rule_based_classifier(rules, data_point)
        if predicted_label is not None:
            covered_count += 1
            if predicted_label == row['class']:
                correctly_classified_count += 1
    if covered_count == 0:
        return 0  # To handle cases where there are no covered data points
    toughness = correctly_classified_count / covered_count
    return toughness



#assignment 5
@api_view(['POST'])
def stats5(request):
    req = json.loads(request.body)
    #for knn classifier
    k = int(req['att2'])
    if (req['att1'] == 'Breast Cancer Dataset'):
        data = datasets.load_breast_cancer()
        df = pd.DataFrame(data=np.c_[data['data'], data['target']], columns=np.append(data['feature_names'], 'target'))

        #Regression
        
        # Select features and target variable
        X = df[['mean radius']]  # Use 'mean radius' as the feature
        y = df['mean area']       # Predict 'mean area'
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # Initialize and train a Linear Regression model
        regressor = LinearRegression()
        regressor.fit(X_train, y_train)
        # Make predictions on the test data
        y_pred = regressor.predict(X_test)
        #evaluation for regression
        mse = mean_squared_error(y_test, y_pred),
        rmse = np.sqrt(mean_squared_error(y_test, y_pred)),
        mae = mean_absolute_error(y_test, y_pred),
        r2 = r2_score(y_test, y_pred)
        #Bayes Classifier
        
        Xb = pd.DataFrame(data.data, columns=data.feature_names)
        yb = pd.Series(data.target)
        Xb_train, Xb_test, yb_train, yb_test = train_test_split(Xb, yb, test_size=0.2, random_state=42)
        classifier = GaussianNB()
        classifier.fit(Xb_train, yb_train)
        yb_pred = classifier.predict(Xb_test)
        accuracy = accuracy_score(yb_test, yb_pred)
        print("Accuracy:", accuracy)
        class_reportB = classification_report(yb_test, yb_pred)
        print("Breast Cancer Dataset Bayes Classifier report :- ")
        print(class_reportB)
        cmB = confusion_matrix(yb_test, yb_pred)
        
        accB = accuracy_score(yb_test, yb_pred) # tp + tn/p+n
        misB = 1 - accB  # fp + fn/p+n
        sensB = recall_score(yb_test, yb_pred,average='macro')   # tp / tp + fn  true positivity scor
        specB = cmB[0, 0] / (cmB[0, 0] + cmB[0, 1])  # tn / tn + fp  true negativity score
        precB = precision_score(yb_test, yb_pred, average='macro')  # tp / tp + fp Positive Predictive Value
        recB = sensB  # Recall is the same as sensitivity

        #k-nn Classifier
        scaler = StandardScaler()
        Xb_train = scaler.fit_transform(Xb_train)
        Xb_test = scaler.transform(Xb_test)
        knn = KNeighborsClassifier(k)
        knn = KNNClassifier(k)
        distances = []
        # distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]
        for x1 in Xb_test :
            for x in Xb_train :
                distances.append(np.sqrt(np.sum((x1 - x) ** 2)))
             # Sort by distance and return indices of the first k neighbors
            k_indices = np.argsort(distances)[k]
            # Extract the labels of the k nearest neighbor training samples
            k_nearest_labels = [yb_train[i] for i in k_indices]
            print("k nearest labels ", k_nearest_labels)
            # Return the most common class label
            most_common = np.bincount(k_nearest_labels).argmax()
            print("most common ", most_common)
            
        knn.fit(Xb_train, yb_train)
        yk_pred = knn.predict(Xb_test)
        #evaluation
        class_reportk = classification_report(yb_test, yk_pred)
        print("Breat Cancer Dataset {k}-NN Classifier report :- ")
        print(class_reportk)
        cmK = confusion_matrix(yb_test, yk_pred)
    
        accK = accuracy_score(yb_test, yk_pred) # tp + tn/p+n
        misK = 1 - accK  # fp + fn/p+n
        sensK = recall_score(yb_test, yk_pred,average='macro')   # tp / tp + fn  true positivity scor
        specK = cmK[0, 0] / (cmK[0, 0] + cmK[0, 1])  # tn / tn + fp  true negativity score
        precK = precision_score(yb_test, yk_pred, average='macro')  # tp / tp + fp Positive Predictive Value
        recK = sensK  # Recall is the same as sensitivity

        #ANN Classifier
        
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(units=16, activation='relu', input_dim=Xb_train.shape[1]),
            tf.keras.layers.Dense(units=8, activation='relu'),
            tf.keras.layers.Dense(units=1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        epochs = 50  # You can adjust the number of training epochs
        batch_size = 32
        model.fit(Xb_train, yb_train, epochs=epochs, batch_size=batch_size, verbose=1)
        ya_pred = (model.predict(Xb_test) > 0.5).astype(int).flatten()  # Convert probabilities to binary labels
        accA = accuracy_score(yb_test, ya_pred)
        error_data = {'epoch': [], 'loss': []}

        for epoch in range(epochs):  # Replace 'epochs' with the actual number of training epochs
            # Collect the loss
            loss = model.evaluate(Xb_train, yb_train)  # Replace with your evaluation method
            
            # Append the data to the error_data dictionary
            error_data['epoch'].append(epoch)
            error_data['loss'].append(loss)
            
        print("Accuracy of ANN of Breast Cancer dataset:", accA)
        # Other metrics
        print("Breast Cancer Dataset ANN Classifier report :- ")
        print("Error graph:- ", )
        print(classification_report(yb_test, ya_pred))
        print(confusion_matrix(yb_test, ya_pred))
        cmA = confusion_matrix(yb_test, ya_pred)
        accA = accuracy_score(yb_test, ya_pred) # tp + tn/p+n
        misA = 1 - accA  # fp + fn/p+n
        sensA = recall_score(yb_test, ya_pred,average='macro')   # tp / tp + fn  true positivity scor
        specA = cmA[0, 0] / (cmA[0, 0] + cmA[0, 1])  # tn / tn + fp  true negativity score
        precA = precision_score(yb_test, ya_pred, average='macro')  # tp / tp + fp Positive Predictive Value
        recA = sensA  # Recall is the same as sensitivity
        
        return Response({
            'mse' : mse,
            'rmse' : rmse,
            'mae' : mae,
            'r2' : r2,
            'accB' : accB,
            'misB' : misB,
            'sensB' : sensB,
            'specB' : specB,
            'precB' : precB,
            'recB' : recB,
            # 'cmB' : cmB,
            'accK' : accK,
            'misK' : misK,
            'sensK' : sensK,
            'specK' : specK,
            'precK' : precK,
            'recK' : recK,
            # 'cmK' : cmK,
            'accA' : accA,
            'misA' : misA,
            'sensA' : sensA,
            'specA' : specA,
            'precA' : precA,
            'recA' : recA,
            'loss' : error_data['loss'],
            'epochs' : error_data['epoch'],
            # 'cmA' : cmA
            
        }, status=status.HTTP_200_OK)

        
    elif (req['att1'] == 'Iris Dataset'):
        iris = datasets.load_iris()
        iris_df = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['target'])
        
        #Regression

        X = iris_df[['sepal length (cm)']]  # Use 'sepal length' as the feature
        y = iris_df['petal length (cm)']    # Predict 'petal length'
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # Initialize and train a Linear Regression model
        regressor = LinearRegression()
        regressor.fit(X_train, y_train)
        # Make predictions on the test data
        y_pred = regressor.predict(X_test)
        #evaluation for regression
        mse = mean_squared_error(y_test, y_pred),
        rmse = np.sqrt(mean_squared_error(y_test, y_pred)),
        mae = mean_absolute_error(y_test, y_pred),
        r2 = r2_score(y_test, y_pred)
        
        #Naive Bayes Classifier
        
        Xb = iris.data
        yb = iris.target
        Xb_train, Xb_test, yb_train, yb_test = train_test_split(Xb, yb, test_size=0.3, random_state=42)
        
        gnb = GaussianNB()
        gnb.fit(Xb_train, yb_train)
        
        yb_pred = gnb.predict(Xb_test)
        #evaluation
        class_reportB = classification_report(yb_test, yb_pred)
        print("Iris Dataset Bayes Classifier report :- ")
        print(class_reportB)
        cmB = confusion_matrix(yb_test, yb_pred)
        
        accB = accuracy_score(yb_test, yb_pred) # tp + tn/p+n
        misB = 1 - accB  # fp + fn/p+n
        sensB = recall_score(yb_test, yb_pred,average='macro')   # tp / tp + fn  true positivity scor
        specB = cmB[0, 0] / (cmB[0, 0] + cmB[0, 1])  # tn / tn + fp  true negativity score
        precB = precision_score(yb_test, yb_pred, average='macro')  # tp / tp + fp Positive Predictive Value
        recB = sensB  # Recall is the same as sensitivity

        #k-nn Classifier
        
        # Split the dataset into training and testing sets
        Xk_train, Xk_test, yk_train, yk_test = train_test_split(Xb, yb, test_size=0.3, random_state=42)
        # knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
        knn = KNNClassifier(k)
        knn.fit(Xk_train, yk_train)
        yk_pred = knn.predict(Xk_test)
        #evaluation
        # Perform 5-fold cross-validation
        scores = cross_val_score(knn, Xb, yb, cv=5)
        
        # Calculate and print the mean accuracy for each k
        mean_accuracy = scores.mean()
        print(f'k = {k}, Mean Accuracy: {mean_accuracy:.2f}')
        class_reportk = classification_report(yk_test, yk_pred)
        print("Iris Dataset K-NN Classifier report :- ")
        print(class_reportk)
        cmK = confusion_matrix(yk_test, yk_pred)
    
        accK = accuracy_score(yk_test, yk_pred) # tp + tn/p+n
        misK = 1 - accK  # fp + fn/p+n
        sensK = recall_score(yk_test, yk_pred,average='macro')   # tp / tp + fn  true positivity scor
        specK = cmK[0, 0] / (cmK[0, 0] + cmK[0, 1])  # tn / tn + fp  true negativity score
        precK = precision_score(yk_test, yk_pred, average='macro')  # tp / tp + fp Positive Predictive Value
        recK = sensK  # Recall is the same as sensitivity

        #ANN Classifier
        
        Xa_train, Xa_test, ya_train, ya_test = train_test_split(Xb, yb, test_size=0.2, random_state=42)

        # Standardize the feature values (optional but recommended)
        scaler = StandardScaler()
        Xa_train = scaler.fit_transform(Xa_train)
        Xa_test = scaler.transform(Xa_test)
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(units=16, activation='relu', input_dim=Xa_train.shape[1]),
            tf.keras.layers.Dense(units=8, activation='relu'),
            tf.keras.layers.Dense(units=1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        epochs = 50  # You can adjust the number of training epochs
        batch_size = 32
        model.fit(Xa_train, ya_train, epochs=epochs, batch_size=batch_size, verbose=1)
        ya_pred = (model.predict(Xa_test) > 0.5).astype(int).flatten()  # Convert probabilities to binary labels
        accA = accuracy_score(ya_test, ya_pred)
        print("Accuracy of ANN of Breast Cancer dataset:", accA)
        # Other metrics
        print("Breast Cancer Dataset ANN Classifier report :- ")
        print(classification_report(ya_test, ya_pred))
        print(confusion_matrix(ya_test, ya_pred))
        cmA = confusion_matrix(ya_test, ya_pred)
        accA = accuracy_score(ya_test, ya_pred) # tp + tn/p+n
        misA = 1 - accA  # fp + fn/p+n
        sensA = recall_score(ya_test, ya_pred,average='macro')   # tp / tp + fn  true positivity scor
        specA = cmA[0, 0] / (cmA[0, 0] + cmA[0, 1])  # tn / tn + fp  true negativity score
        precA = precision_score(ya_test, ya_pred, average='macro')  # tp / tp + fp Positive Predictive Value
        recA = sensA  # Recall is the same as sensitivity
        
        
        return Response({
            'mse' : mse,
            'rmse' : rmse,
            'mae' : mae,
            'r2' : r2,
            'accB' : accB,
            'misB' : misB,
            'sensB' : sensB,
            'specB' : specB,
            'precB' : precB,
            'recB' : recB,
            'accK' : accK,
            'misK' : misK,
            'sensK' : sensK,
            'specK' : specK,
            'precK' : precK,
            'recK' : recK,
            'accA' : accA,
            'misA' : misA,
            'sensA' : sensA,
            'specA' : specA,
            'precA' : precA,
            'recA' : recA,
            
        }, status=status.HTTP_200_OK)



