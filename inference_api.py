'''
GenresClassifier class - loads the trained model and makes probabilistic predictions,
given the plot & the threshold (optional)

'''

import preprocessor
from model_train import ModelTrain

DEFAULT_THRESHOLD = 0.2


class GenresClassifier:

    # load model from classifier_api
    def __init__(self):
        classifier_api = ModelTrain()
        loaded = classifier_api.load_model()
        if loaded:
            self.model = loaded["model"]
            self.vectorizer = loaded["vectorizer"]
            self.binarizer = loaded["binarizer"]
        else:
            raise Exception("Model couldn't be loaded")

    # the endpoint for the service - predicts genres based on given plot and threshold
    def predict_genres(self, plot, threshold=DEFAULT_THRESHOLD):

        # do the same preprocessing as was done in training
        text = preprocessor.clean_text(plot)
        text = [text]

        # vectorize using the saved in training vectorizer
        x_tfidf = self.vectorizer.transform(text)

        # predict with probabilities
        y_pred_prob = self.model.predict_proba(x_tfidf)

        # fill the results with [saved in binorizer] classes predicted with probability over the threshold
        results = []
        for idx in range(len(y_pred_prob)):
            for cl, prob in zip(self.binarizer.classes_, y_pred_prob[idx]):
                if prob >= threshold:
                    results.append({cl: "{:.2f}".format(prob)})

        # hard prediction - using default threshold = 0.5
        # prediction = self.model.predict(x_tfidf)
        # results = self.multilabel_binarizer.inverse_transform(prediction)

        return results

