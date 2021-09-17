'''
orchestrator.py - orchestrates the training, evaluation, and inference for sanity checks
'''
from data_loader import DataLoader
from model_train import ModelTrain
from inference_api import GenresClassifier

FEATURE_COLUMN = 'plot_summary'
LABELS_COLUMN = 'genres'


def get_data():
    filename = 'train_reduced.csv'
    df = DataLoader.read_csv(filename, LABELS_COLUMN)
    return df


def evaluate_algorithms(data):
    classifier_api = ModelTrain()
    classifier_api.train_and_evaluate(data, text_column=FEATURE_COLUMN, target_column=LABELS_COLUMN)


def train_and_save_model(data):
    classifier_api = ModelTrain()
    clf = classifier_api.train_and_save(data, text_column=FEATURE_COLUMN, target_column=LABELS_COLUMN)
    return clf


def classify(text):
    inference_api = GenresClassifier()
    return inference_api.predict_genres(text)


if __name__ == '__main__':
    # train_data = get_data()

    # evaluate_algorithms(train_data)
    # train_and_save_model(train_data)
    # exit()

    sample_plot = "Four friends- Gangu , Abdul , Nihal  and Gary - get together to start their business, " \
                  "but their roots are built on friendship and trust. They succeed in their criminal goals, " \
                  "although Gangu is arrested and sentenced to jail for five years. Before going to jail, " \
                  "he asks them to promise to go straight, to which they all agree. When Gangu is released, " \
                  "he is pleased to find that Abdul is now driving a taxi, his mother is well looked after, " \
                  "and that Nihal and Gary have also started doing business. It is when Gangu meets his sweetheart, " \
                  "Sanam , and proposes marriage, that he learns that all is not well in their world."

    prediction = classify(sample_plot)
    print(prediction)





