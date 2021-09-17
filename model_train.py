'''
class ModelTrain - trains models, evaluates them, keeps the best one;
aux methods for saving/loading
'''
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import hamming_loss, f1_score
from sklearn.metrics import accuracy_score
import os
import logging
import datetime
import joblib

import preprocessor
import utilities

MODEL_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'models')
MODEL_FILENAME = 'finalized_model.sav'
logging.basicConfig(filename='classification_report.log', encoding='utf-8', level=logging.DEBUG)
MIN_DF = 2
MAX_DF = 0.8
MAX_FEATURES = 5000


class ModelTrain:

    @staticmethod
    def save_model(model, vectorizer, binarizer):
        """
        saves model to disk as dict of three entities:
        the model itself, the vectorizer used for text feature, the binarizer for labels
        """
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
        try:
            full_path = os.path.join(MODEL_DIR, MODEL_FILENAME)
            to_dump = {"model": model, "vectorizer": vectorizer, "binarizer": binarizer}
            joblib.dump(to_dump, full_path)
        except:
            logging.error(f'Saving model to {full_path} failed')

    @staticmethod
    def load_model():
        """
        loads model from disk, returns the whole dict, w/o unpacking (processed outside)
        """
        try:
            full_path = os.path.join(MODEL_DIR, MODEL_FILENAME)
            model = joblib.load(full_path)
        except:
            return None
        return model

    @staticmethod
    def train_and_evaluate(data, text_column, target_column):
        """
        gets training data and runs different regression algorithms with different threshold values,
        evaluating each one and logging the results
        """
        # preprocess text feature
        X = preprocessor.process_text_column(data[text_column])

        # transform labels
        multilabel_binarizer = MultiLabelBinarizer()
        labels = data[target_column]
        multilabel_binarizer.fit(labels)
        y = multilabel_binarizer.transform(labels)

        xtrain, xval, ytrain, yval = train_test_split(X, y, test_size=0.2, random_state=42)

        # vectorize text, both train and validation
        tfidf_vectorizer = TfidfVectorizer(min_df=MIN_DF, max_df=MAX_DF, max_features=MAX_FEATURES)
        xtrain_tfidf = tfidf_vectorizer.fit_transform(xtrain)
        xval_tfidf = tfidf_vectorizer.transform(xval)

        # algorithm candidates
        lr = LogisticRegression(solver='liblinear')
        nb_clf = MultinomialNB()
        sgd = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=6, tol=None)

        classifiers = {"lr": lr, "nb": nb_clf, "sgd": sgd}

        # iterate over algorithms and evaluate
        for key, value in classifiers.items():

            clf = OneVsRestClassifier(value)

            # perform grid search
            # param_grid = {"estimator__C": np.logspace(-3, 3, 5), "estimator__penalty": ["l1", "l2"]}
            # CV_clf = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5)

            # train
            logging.info(f'Start train at {datetime.datetime.now()}')
            best_clf = clf.fit(xtrain_tfidf, ytrain)
            logging.info(f'End train at {datetime.datetime.now()}')

            # make predictions for validation set
            y_pred = clf.predict(xval_tfidf)

            # calculate metrics
            f1 = f1_score(yval, y_pred, average="micro")
            loss = hamming_loss(yval, y_pred)
            score = utilities.hamming_score(yval, y_pred)

            logging.info(f'classifier: {key}')
            logging.info(f'loss = {loss}\n'
                         f'f1={f1}\n'
                         f'hamming_score={score}\n')

            # try probabilistic predictions with threshold from 0.1 to 0.6 with step of 0.1

            # probabilities are not availble for hinge loss which we use in SGD
            if key == 'sgd':
                continue

            y_pred_prob = clf.predict_proba(xval_tfidf)

            for t in np.arange(0.1, 0.7, 0.1):
                y_pred_new = (y_pred_prob >= t).astype(int)
                f1 = f1_score(yval, y_pred_new, average="micro")
                score = utilities.hamming_score(yval, y_pred_new)

                # log the metrics
                logging.info(f'\nnew threshold: {t}')
                logging.info(f'f1={f1}\nhamming_score={score}\n')

    def train_and_save(self, data, text_column, target_column):
        # preprocess text
        X = preprocessor.process_text_column(data[text_column])

        # transform labels y
        multilabel_binarizer = MultiLabelBinarizer()
        labels = data[target_column]
        multilabel_binarizer.fit(labels)
        y = multilabel_binarizer.transform(labels)

        # vectorize text X
        tfidf_vectorizer = TfidfVectorizer(min_df=MIN_DF, max_df=MAX_DF, max_features=MAX_FEATURES)
        xtrain_tfidf = tfidf_vectorizer.fit_transform(X)

        lr = LogisticRegression(solver='liblinear')
        clf = OneVsRestClassifier(lr)

        # train
        logging.info(f'Start train at {datetime.datetime.now()}')
        model = clf.fit(xtrain_tfidf, y)
        logging.info(f'End train at {datetime.datetime.now()}')

        # save the model
        self.save_model(model, tfidf_vectorizer, multilabel_binarizer)
        logging.info(f'Model saved')

        return model
