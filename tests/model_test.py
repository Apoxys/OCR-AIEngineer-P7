import unittest
import pickle
import os
import numpy as np

class TestRegressionModel(unittest.TestCase):
    MODEL_PATH = "model_to_keep/tf_idf_LR/tf_idf_LR_model.pkl"
    VECTORIZER_PATH = "model_to_keep/tf_idf_LR/tf_idfvectorizer.pkl"
    VOCAB = "model_to_keep/tf_idf_LR/tfidf_vocab.json"
    TEST_SENTENCE = "This was amazing."

    def setUp(self):
        #Loading model
        if not os.path.exists(self.MODEL_PATH) or not os.path.exists(self.VECTORIZER_PATH):
            self.fail("Model, vectorizer or vocabulary is missing from path. Check path again")

        with open(self.MODEL_PATH, 'rb') as model_file:
            self.model = pickle.load(model_file)

        with open(self.VECTORIZER_PATH, 'rb') as vectorizer_file:
            self.vectorizer = pickle.load(vectorizer_file)

    def test_prediction_consistency(self):
        #checking predictions stability
        transformed_sentence = self.vectorizer.transform([self.TEST_SENTENCE])
        predictions = np.array([self.model.predict(transformed_sentence)[0] for _ in range(5)])

        self.assertTrue(np.all(predictions == predictions[0]), 
                        "unstable predictions")

if __name__ == "__main__":
    unittest.main()
