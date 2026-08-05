from collections import Counter
from sys import displayhook
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from statistics import mode
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def get_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data()

    word_index = tf.keras.datasets.imdb.get_word_index()
    index2word = dict((i + 3, word) for (word, i) in word_index.items())

    index2word[0] = '[pad]'
    index2word[1] = '[bos]'
    index2word[2] = '[oov]'

    x_train = np.array([' '.join([index2word[idx] for idx in text]) for text in x_train])
    x_test = np.array([' '.join([index2word[idx] for idx in text]) for text in x_test])

    return x_train, y_train, x_test, y_test

def binary_representation(x_train, x_test, vocab):
    binary_vectorizer = CountVectorizer(binary=True, vocabulary = vocab.keys())

    x_train_binary = binary_vectorizer.fit_transform(x_train)
    x_train_binary = x_train_binary.toarray()

    x_test_binary = binary_vectorizer.transform(x_test)
    x_test_binary = x_test_binary.toarray()
    
    #print('Vocabulary size:', len(binary_vectorizer.vocabulary_))
    return x_train_binary, x_test_binary

def create_vocab(x_train, n, k, m):

    all_words = [word for text in x_train for word in set(text.split())]
    word_frequency = dict(Counter(all_words))

    word_frequency.pop('[pad]', None)
    word_frequency.pop('[bos]', None)
    word_frequency.pop('[oov]', None)

    vocab = dict(sorted(word_frequency.items(), key=lambda item: item[1], reverse=True))
   
    vocab = dict(list(vocab.items())[n:-k][:m])

    return vocab


#Random Forest
class RandomForest:
    def __init__(self, n_trees):
        self.n_trees = n_trees
        self.trees = []

    def fit(self, x_train, y_train):
        for _ in range(self.n_trees):

            #get a subset of the training data
            indices = np.random.choice(len(x_train), size=len(x_train), replace=True)
            subset_features = np.random.choice(x_train.shape[1], size=2000, replace=True)
            
            subset_x_train = x_train[indices][:, subset_features]
            subset_y_train = y_train[indices]

            tree1 = ID3(features=subset_features)
            root1 = tree1.fit(x=subset_x_train, y=subset_y_train, depth=0)
            self.trees.append(root1)

    def predict(self, x):
        predicted_classes = list()

        for unlabeled in x:  # for every example 
            predictions = list()

            for tmp in self.trees:  # for every tree
                while not tmp.is_leaf:
                    if unlabeled.flatten()[tmp.checking_feature] == 1:
                        tmp = tmp.left_child
                    else:
                        tmp = tmp.right_child

                predictions.append(tmp.category)

            predicted_classes.append(mode(predictions))
        
        return np.array(predicted_classes)

class Node:
    def __init__(self, checking_feature=None, is_leaf=False, category=None):
        self.checking_feature = checking_feature
        self.left_child = None
        self.right_child = None
        self.is_leaf = is_leaf
        self.category = category
        
class ID3:
    def __init__(self, features):
        self.tree = None
        self.features = features
        
    def fit(self, x, y, depth=0):
        
        #creates the tree

        most_common = mode(y.flatten())
        
        self.tree = self.create_tree(x, y, features=np.arange(len(self.features)), category=most_common, depth=depth)
        return self.tree
        
    def create_tree(self, x_train, y_train, features, category, depth):
        
        # check empty data
        if len(x_train) == 0:
            return Node(checking_feature=None, is_leaf=True, category=category)  # decision node
            
        # check all examples belonging in one category
        if np.all(y_train.flatten() == 0):
            return Node(checking_feature=None, is_leaf=True, category=0)
        elif np.all(y_train.flatten() == 1):
            return Node(checking_feature=None, is_leaf=True, category=1)
            
        if len(features) == 0:
            return Node(checking_feature=None, is_leaf=True, category=mode(y_train.flatten()))
        
        if depth >= 3:
            return Node(checking_feature=None, is_leaf=True, category=mode(y_train.flatten()))
            
        igs = list()
        for feat_index in features.flatten():
            igs.append(self.calculate_ig(y_train.flatten(), [example[feat_index] for example in x_train]))
            
        max_ig_idx = np.argmax(np.array(igs).flatten())
        m = mode(y_train.flatten())  # most common category 

        root = Node(checking_feature=max_ig_idx)

        # data subset with X = 0
        x_train_0 = x_train[x_train[:, max_ig_idx] == 0, :]
        y_train_0 = y_train[x_train[:, max_ig_idx] == 0].flatten()
        
        # data subset with X = 1
        x_train_1 = x_train[x_train[:, max_ig_idx] == 1, :]
        y_train_1 = y_train[x_train[:, max_ig_idx] == 1].flatten()

        new_features_indices = np.delete(features.flatten(), max_ig_idx)  # remove current feature
        
        root.left_child = self.create_tree(x_train=x_train_1, y_train=y_train_1, features=new_features_indices, category=m, depth=depth + 1)  # go left for X = 1
            
        root.right_child = self.create_tree(x_train=x_train_0, y_train=y_train_0, features=new_features_indices, category=m, depth=depth + 1)  # go right for X = 0
            
        return root
        
    @staticmethod
    def calculate_ig(classes_vector, feature):
        classes = set(classes_vector)

        HC = 0
        for c in classes:
            PC = list(classes_vector).count(c) / len(classes_vector)  # P(C=c)
            HC += - PC * math.log(PC, 2)  # H(C)
            # print('Overall Entropy:', HC)  # entropy for C variable
                
        feature_values = set(feature)  # 0 or 1 in this example
        HC_feature = 0
        for value in feature_values:
            # pf --> P(X=x)
            pf = list(feature).count(value) / len(feature)  # count occurences of value 
            indices = [i for i in range(len(feature)) if feature[i] == value]  # rows (examples) that have X=x

            classes_of_feat = [classes_vector[i] for i in indices]  # category of examples listed in indices above
            for c in classes:
                # pcf --> P(C=c|X=x)
                pcf = classes_of_feat.count(c) / len(classes_of_feat)  # given X=x, count C
                if pcf != 0: 
                    # - P(X=x) * P(C=c|X=x) * log2(P(C=c|X=x))
                    temp_H = - pf * pcf * math.log(pcf, 2)
                    # sum for all values of C (class) and X (values of specific feature)
                    HC_feature += temp_H
        
        ig = HC - HC_feature
        return ig    




#get data / create vocabulary / binary representation

x_train_imdb, y_train_imdb, x_test_imdb, y_test_imdb = get_data()
vocabulary = create_vocab(x_train = x_train_imdb, n = 50, k = 85000, m = 2500)
x_train_imdb_binary, x_test_imdb_binary = binary_representation(x_train = x_train_imdb, x_test = x_test_imdb, vocab = vocabulary)




#Train/Predict RandomForest
rf = RandomForest(n_trees=10)
rf.fit(x_train_imdb_binary, y_train_imdb)
y = rf.predict(x_test_imdb_binary)
print(classification_report(y_test_imdb, y, zero_division=1))

#Train/Predict RandomForestClassifier
rf = RandomForestClassifier(criterion='entropy')
rf.fit(x_train_imdb_binary, y_train_imdb)
y = rf.predict(x_test_imdb_binary)
print(classification_report(y_test_imdb, y, zero_division=1))





def scores(model, x_train, y_train, x_test, y_test, step_size):
    train_sizes = list(range(step_size, len(x_train) + 1, step_size))
    
    train_accuracy = list()
    test_accuracy = list()

    train_precision = list()
    test_precision = list()

    train_recall = list()
    test_recall = list()

    train_f1 = list()
    test_f1 = list()

    for size in train_sizes:
        model.fit(x_train[:size], y_train[:size])

        # Calculating accuracy on training data
        train_pred = model.predict(x_train[:size])

        train_accuracy.append(accuracy_score(y_train[:size], train_pred))
        train_precision.append(precision_score(y_train[:size], train_pred))
        train_recall.append(recall_score(y_train[:size], train_pred))
        train_f1.append(f1_score(y_train[:size], train_pred))

        # Calculating accuracy in control data
        test_pred = model.predict(x_test)

        test_accuracy.append(accuracy_score(y_test, test_pred))
        test_precision.append(precision_score(y_test, test_pred))
        test_recall.append(recall_score(y_test, test_pred))
        test_f1.append(f1_score(y_test, test_pred))

    return train_sizes, train_accuracy, test_accuracy, train_precision, test_precision, train_recall, test_recall, train_f1, test_f1

def create_curve(train_sizes, train_score, test_score, type):
    plt.figure()
    plt.plot(train_sizes, train_score, 'o-', color="b", label='Train')
    plt.plot(train_sizes, test_score, 'o-', color="red", label='Test')
    plt.xlabel('Training Size')
    plt.ylabel(type)
    plt.legend()
    plt.show()

def create_table(train_sizes, train_accuracy, test_accuracy, train_precision, test_precision, train_recall, test_recall, train_f1, test_f1):
    table = pd.DataFrame({
        'Training Size': train_sizes,
        'Accuracy Train': train_accuracy,
        'Accuracy Test': test_accuracy,
        'Precision Train': train_precision,
        'Precision Test': test_precision,
        'Recall Train': train_recall,
        'Recall Test': test_recall,
        'F1 Train': train_f1,
        'F1 Test': test_f1
    })

    print(table)





#Curves and table for Random Forest
rf = RandomForest(n_trees=10)
train_sizes, train_accuracy, test_accuracy, train_precision, test_precision, train_recall, test_recall, train_f1, test_f1 = scores(rf, x_train_imdb_binary, y_train_imdb, x_test_imdb_binary, y_test_imdb, step_size=5000)
create_table(train_sizes, train_accuracy, test_accuracy, train_precision, test_precision, train_recall, test_recall, train_f1, test_f1)

create_curve(train_sizes, train_accuracy, test_accuracy, 'Accuracy')
create_curve(train_sizes, train_precision, test_precision, 'Precision')
create_curve(train_sizes, train_recall, test_recall, 'Recall')
create_curve(train_sizes, train_f1, test_f1, 'F1')


#Curves and table for Random Forest Classifier
rf = RandomForestClassifier()
train_sizes, train_accuracy, test_accuracy, train_precision, test_precision, train_recall, test_recall, train_f1, test_f1 = scores(rf, x_train_imdb_binary, y_train_imdb, x_test_imdb_binary, y_test_imdb, step_size=5000)
create_table(train_sizes, train_accuracy, test_accuracy, train_precision, test_precision, train_recall, test_recall, train_f1, test_f1)

create_curve(train_sizes, train_accuracy, test_accuracy, 'Accuracy')
create_curve(train_sizes, train_precision, test_precision, 'Precision')
create_curve(train_sizes, train_recall, test_recall, 'Recall')
create_curve(train_sizes, train_f1, test_f1, 'F1')

