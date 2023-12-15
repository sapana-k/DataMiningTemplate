from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
import json
from django.http import HttpResponse,JsonResponse, FileResponse
from rest_framework import status
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statistics as st
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

@api_view(['POST'])
def upload_file(request):
    file = request.FILES['file']
    dfd = pd.read_csv(file)
    return Response({'data':dfd}, status=status.HTTP_200_OK)

#apriori
@api_view(['POST'])
def apriorii(request):
    req = json.loads(request.body)
    itemsets = {}
    url = "C:\\Users\\sapan\\Downloads\\adminPanelAssignment1\\dataset\\groceries - groceries.csv"
    # Load the dataset into a pandas DataFrame
    bakery_df = pd.read_csv(url)
    # Specify the column to exclude
    column_to_exclude = 'Item(s)'
    # Access all columns except the specified one
    selected_columns = bakery_df.drop(column_to_exclude, axis=1)
    # candidateItemset2 = selected_columns["Item 2"].value_counts()
    
    #finding support for each item in each column
    total1Itemset = dict()
    for x in selected_columns:
        CandidateItemset1 = dict()
        for y in selected_columns[x]:
            CandidateItemset1[y] = CandidateItemset1.get(y, 0) + 1
        total1Itemset[x] = CandidateItemset1
        
    CandidateItem1set1 = dict()
    for y in selected_columns["Item 1"]:
        CandidateItem1set1[y] = CandidateItem1set1.get(y, 0) + 1
    
    CandidateItem2set1 = dict()
    for y in selected_columns["Item 2"]:
        CandidateItem2set1[y] = CandidateItem2set1.get(y, 0) + 1    
    #for 2-itemset i am choosing column Item 1 and Item 2
    #find count of each pair
    pair_counts = dict()
    for i in range(len(selected_columns["Item 1"])):
        if selected_columns["Item 1"][i] != selected_columns["Item 2"][i]:
            pair = (selected_columns["Item 1"][i], selected_columns["Item 2"][i])
            pair_counts[pair] = pair_counts.get(pair, 0) + 1
    
    # for pair, count in pair_counts.items():
    #     print(f'{pair}: {count}')       
    #generate frequent itemsets
    min_support = float(req['att1'])
    min_confidence = float(req['att2']) 
    
    frequent1Itemset = dict()
    for item, count in CandidateItemset1.items():
        if(count/len(selected_columns["Item 1"]) >= min_support):
            frequent1Itemset[item] = count
            
    frequent2Itemset = dict()
    for item_pair, count in pair_counts.items():
        if(count/len(selected_columns["Item 1"]) >= min_support):
            frequent2Itemset[item_pair] = count
    
    #generate association rules with frequent itemsets 
    association_rules = []
    for item_pair, count in frequent2Itemset.items():
        antecedent_support = CandidateItem1set1[item_pair[0]]
        if not pd.isna(item_pair[1]):
            consequent_support = CandidateItem1set1[item_pair[1]]
            rule_support = pair_counts[item_pair]
            confidence = count/antecedent_support
            expected_support = antecedent_support * consequent_support
            precision = rule_support / antecedent_support
            recall = rule_support / consequent_support
            
            lift = rule_support / expected_support
            all_confidence = rule_support / antecedent_support
            max_confidence = rule_support / min(antecedent_support, consequent_support)
            kulczynski = (precision + recall) / 2
            cosine_measure = rule_support / (antecedent_support * consequent_support) ** 0.5
            
            if confidence >= min_confidence:
                association_rules.append((item_pair, [confidence, lift, all_confidence, max_confidence, kulczynski, cosine_measure])) 
    print(association_rules)
    nan_safe_data = []
    for (item1, item2), value in association_rules:
        # if pd.isna(item2):
        #     nan_safe_data.append(((item1, None), value))
        # else:
        #     nan_safe_data.append(((item1, item2), round(value, 3)))
        for i in range(len(value)):
            value[i] = round(value[i], 4)
        nan_safe_data.append(((item1, item2), value))
        # Sort the list based on the second element of each tuple
        sorted_data = sorted(nan_safe_data, key=lambda x: x[1][0], reverse=True)
        
    return Response({'rules':sorted_data}, status = status.HTTP_200_OK)








































# class Apriori:
#     def __init__(self, transactions, min_support, min_confidence):
#         self.transactions = transactions
#         self.min_support = min_support
#         self.min_confidence = min_confidence
#         self.itemsets = {}
#         self.rules = []

#     def generate_frequent_itemsets(self):
#         self.itemsets[1] = self.get_frequent_1_itemsets()

#         k = 2
#         while self.itemsets[k - 1]:
#             candidate_itemsets = self.generate_candidate_itemsets(self.itemsets[k - 1], k)
#             frequent_itemsets = self.get_frequent_itemsets(candidate_itemsets)
#             self.itemsets[k] = frequent_itemsets
#             k += 1

#     def get_frequent_1_itemsets(self):
#         unique_items = set(item for transaction in self.transactions for item in transaction)
#         frequent_itemsets = {frozenset([item]): 0 for item in unique_items}

#         for transaction in self.transactions:
#             for itemset in frequent_itemsets.keys():
#                 if itemset.issubset(transaction):
#                     frequent_itemsets[itemset] += 1

#         return {itemset: support for itemset, support in frequent_itemsets.items() if support >= self.min_support}

#     def generate_candidate_itemsets(self, prev_itemsets, k):
#         candidates = set()
#         for itemset1 in prev_itemsets:
#             for itemset2 in prev_itemsets:
#                 union_set = itemset1.union(itemset2)
#                 if len(union_set) == k and itemset1 != itemset2:
#                     candidates.add(union_set)
#         return candidates

#     def get_frequent_itemsets(self, candidate_itemsets):
#         frequent_itemsets = {}
#         for transaction in self.transactions:
#             for candidate in candidate_itemsets:
#                 if candidate.issubset(transaction):
#                     if candidate in frequent_itemsets:
#                         frequent_itemsets[candidate] += 1
#                     else:
#                         frequent_itemsets[candidate] = 1

#         return {itemset: support for itemset, support in frequent_itemsets.items() if support >= self.min_support}

#     def generate_association_rules(self):
#         for k, itemsets in self.itemsets.items():
#             if k > 1:
#                 for itemset in itemsets:
#                     self.generate_rules(itemset)

#     def generate_rules(self, itemset):
#         subsets = self.get_subsets(itemset)

#         for subset in subsets:
#             confidence = self.itemsets[len(subset)][subset] / self.itemsets[len(itemset) - len(subset)][subset]
#             if confidence >= self.min_confidence:
#                 rule = (set(subset), set(itemset - subset), confidence)
#                 self.rules.append(rule)

#     def get_subsets(self, itemset):
#         subsets = []
#         for i in range(1, len(itemset)):
#             subsets.extend(combinations(itemset, i))
#         return subsets

# if __name__ == "__main__":
#     # Sample UCI Bakery Dataset
#     url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00373/BreadBasket_DMS.csv"
#     # Load the dataset into a pandas DataFrame
#     bakery_df = pd.read_csv(url)
    
#     transactions = [
#         {"Bread", "Milk"},
#         {"Bread", "Diapers", "Beer", "Eggs"},
#         {"Milk", "Diapers", "Beer", "Coke"},
#         {"Bread", "Milk", "Diapers", "Beer"},
#         {"Bread", "Milk", "Diapers", "Coke"}
#     ]

#     min_support = 2
#     min_confidence = 0.6

#     apriori = Apriori(transactions, min_support, min_confidence)
#     apriori.generate_frequent_itemsets()
#     apriori.generate_association_rules()

#     print("Frequent Itemsets:")
#     for k, itemsets in apriori.itemsets.items():
#         print(f"Itemsets of size {k}: {itemsets}")

#     print("\nAssociation Rules:")
#     for rule in apriori.rules:
#         print(f"Rule: {rule[0]} => {rule[1]}, Confidence: {rule[2]:.2f}")
