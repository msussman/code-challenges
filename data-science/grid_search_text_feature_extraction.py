
"""
==========================================================
Pipeline for text feature extraction and evaluation
==========================================================

The dataset used in this example is the 20 newsgroups dataset which will be
automatically downloaded and then cached and reused for the document
classification example.

You can adjust the number of categories by giving their names to the dataset
loader or setting them to None to get the 20 of them.



"""

# Author: Olivier Grisel <olivier.grisel@ensta.org>
#         Peter Prettenhofer <peter.prettenhofer@gmail.com>
#         Mathieu Blondel <mathieu@mblondel.org>
# License: BSD 3 clause

from __future__ import print_function

from pprint import pformat
import time
import logging

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib


###############################################################################
class TextFeatureExtractor():
    def __init__(self, categories, param_grid):
        self.categories = categories
        self.pipeline = Pipeline([
                                 ('vect', CountVectorizer()),
                                 ('tfidf', TfidfTransformer()),
                                 ('clf', SGDClassifier(max_iter=5)),
                                ])
        self.param_grid = param_grid
        self.data = None
        self.search_results = None
        self.best_parameters = {}
        self.timestr = time.strftime("%Y%m%d_%H%M%S")
        self.init_log()
        self.init_data()
        self.grid_search()
        self.grid_fit()
        self.extract_best_parameters()
        self.pickle_results()

    def init_log(self):
        '''
        Initialize the log
        '''
        logging.basicConfig(filename='logs/grid_search_{}.txt'.format(self.timestr),
                            filemode='w',
                            format='%(asctime)s -  %(levelname)s - %(message)s',
                            level=logging.INFO)

    def init_data(self):
        '''
        Loads 20 newsgroups training dataset for requested categories
        '''
        logging.info("Loading 20 newsgroups dataset for categories:")
        logging.info(self.categories)

        self.data = fetch_20newsgroups(subset='train', 
                                       categories=self.categories)
        logging.info("{} documents".format(len(self.data.filenames)))
        logging.info("{} categories".format(len(self.data.target_names)))
        logging.info("")
        return self.data
    
    def grid_search(self):
        '''
        Performs Grid Search based on passed parameters
        '''
        self.search_results = GridSearchCV(self.pipeline, self.param_grid, 
                                           n_jobs=-1, verbose=1) 
        return self.search_results

    def grid_fit(self):
        '''
        Fits data to the Grid Search parameters
        '''
        logging.info("Performing grid search...")
        logging.info("pipeline:{}".format([name for name, _ in self.pipeline.steps]) )
        logging.info("parameters:")
        logging.info(pformat(self.param_grid))
        t0 = time.time()
        self.search_results.fit(self.data.data, self.data.target)
        logging.info("done in {0:.3f}".format((time.time() - t0)) )
        logging.info("")

    def extract_best_parameters(self):
        '''
        extracts the best parameters from the Grid Search
        '''
        logging.info("Best score: {0:.3f}".format(self.search_results.best_score_))
        logging.info("Best parameters set:")
        self.best_parameters = self.search_results.best_estimator_.get_params()
        for param_name in sorted(self.param_grid.keys()):
            logging.info("\t{}: {}".format(param_name, self.best_parameters[param_name]))
        return self.best_parameters
    
    def pickle_results(self):
        '''
        Pickles the results into timestamped file 
        '''
        pickle_name = 'output/grid_search_parameters_{}.pkl'.format(self.timestr)
        joblib.dump(self.best_parameters, pickle_name, compress = 1)


###############################################################################

# define a pipeline combining a text feature extractor with a simple
# classifier

def main():

    # Define Categories to pull newsfeed training data for, None will pull all 
    categories = ['alt.atheism', 'talk.religion.misc']

    # Pass the grid parameters to use in the search
    param_grid = {
        'vect__max_df': (0.5, 0.75, 1.0),
        #'vect__max_features': (None, 5000, 10000, 50000),
        'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
        #'tfidf__use_idf': (True, False),
        #'tfidf__norm': ('l1', 'l2'),
        'clf__alpha': (0.00001, 0.000001),
        'clf__penalty': ('l2', 'elasticnet'),
        #'clf__n_iter': (10, 50, 80),
    }
    # Run text feature extractor
    TextFeatureExtractor(categories, param_grid)


if __name__ == "__main__":

    main()

