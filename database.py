import json
import pymongo
import pandas as pd
import re
from bson import json_util
from dateutil.parser import parse
import collections
import os


real_mongo_port = 41356
lab_hostname = '2potato'

regression_primitives = [
    'd3m.primitives.regression.ard.SKlearn',
    'd3m.primitives.regression.decision_tree.SKlearn',
    'd3m.primitives.regression.extra_trees.SKlearn',
    'd3m.primitives.regression.gaussian_process.SKlearn',
    'd3m.primitives.regression.gradient_boosting.SKlearn',
    'd3m.primitives.regression.k_neighbors.SKlearn',
    'd3m.primitives.regression.kernel_ridge.SKlearn',
    'd3m.primitives.regression.lars.SKlearn',
    'd3m.primitives.regression.lasso.SKlearn',
    'd3m.primitives.regression.lasso_cv.SKlearn',
    'd3m.primitives.regression.linear_svr.SKlearn',
    'd3m.primitives.regression.mlp.SKlearn',
    'd3m.primitives.regression.passive_aggressive.SKlearn',
    'd3m.primitives.regression.random_forest.SKlearn',
    'd3m.primitives.regression.ridge.SKlearn',
    'd3m.primitives.regression.sgd.SKlearn',
    'd3m.primitives.regression.svr.SKlearn'
]

ensemble_primitives = [
    'd3m.primitives.data_transformation.horizontal_concat.DataFrameConcat',
    'd3m.primitives.data_preprocessing.EnsembleVoting.DSBOX'
]

helper_primitives = [
    'd3m.primitives.data_transformation.column_parser.DataFrameCommon',
    'd3m.primitives.data_transformation.construct_predictions.DataFrameCommon',
    'd3m.primitives.data_transformation.dataset_to_dataframe.Common',
    'd3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon'
]


class DatabaseConnection:

    def __init__(self, hostname=lab_hostname, port=real_mongo_port):
        self.mongo_client = None
        self.connect_to_mongo(hostname, port)

    def connect_to_mongo(self, host_name=lab_hostname, mongo_port=real_mongo_port):
        """
        Connects and returns a session to the mongo database
        :param host_name: the host computer that has the database server
        :param mongo_port: the port number of the database
        :return: a MongoDB session
        """
        try:
            self.mongo_client = pymongo.MongoClient(host_name, mongo_port)
        except Exception as e:
            print("Cannot connect to the Mongo Client at port {}. Error is {}".format(mongo_port, e))

    @staticmethod
    def _get_all(collection, query):
        query = {} if not query else query
        return collection.find(query)

    def _get(self, collection, query):
        query = {} if not query else query
        return collection.find_one(query)

    def get_dataset_docs(self, query=None):
        return self._get_all(self.mongo_client.metalearning.datasets, query)

    def get_problem_docs(self, query=None):
        return self._get_all(self.mongo_client.metalearning.problems, query)

    def get_pipeline_docs(self, query=None):
        return self._get_all(self.mongo_client.metalearning.pipelines, query)

    def get_pipeline_run_docs(self, query=None):
        return self._get_all(self.mongo_client.metalearning.pipeline_runs, query)

    def get_metafeature_docs(self, query=None):
        return self._get_all(self.mongo_client.metalearning.metafeatures, query)

    def get_dataset_doc(self, query=None):
        return self._get(self.mongo_client.metalearning.datasets, query)

    def get_problem_doc(self, query=None):
        return self._get(self.mongo_client.metalearning.problems, query)

    def get_pipeline_doc(self, query=None):
        return self._get(self.mongo_client.metalearning.pipelines, query)

    def get_pipeline_run_doc(self, query=None):
        return self._get(self.mongo_client.metalearning.pipeline_runs, query)

    def get_metafeature_doc(self, query=None):
        return self._get(self.mongo_client.metalearning.metafeatures, query)


if __name__ == "__main__":
    dataset_dir = '/users/data/d3m/datasets'
    dc = DatabaseConnection()
    datasets_data = []
    pipelines_data = []

    pipelines = dc.get_pipeline_docs(query={'steps.primitive.python_path': {'$nin': ensemble_primitives}})
    print(pipelines.count())
    pipe = pipelines[0]
    pipe.pop('_id')
    # print(json.dumps(pipe, indent=2))
    pipeline_digests = [pipe['digest'] for pipe in pipelines]

    runs = dc.get_pipeline_run_docs(query={'pipeline.digest': {'$in': pipeline_digests}})
    run = runs[0]
    run.pop('_id')
    print(json.dumps(run, indent=2))
    for run in runs:
        pipeline_id = run['pipeline']['digest']
        run_id = run['id']
        problem_id = run['problem']['id']

        assert len(run['datasets']) == 1
        dataset_id = run['datasets'][0]['id']

        problem = dc.get_problem_doc(query={'about.problemID': problem_id})
        assert len(problem['inputs']['data'][0]['targets']) == 1
        target = problem['inputs']['data'][0]['targets'][0]['colName']
        problem.pop('_id')
        # print(json.dumps(problem, indent=2))

        dataset = dc.get_dataset_doc(query={'about.datasetID': dataset_id})
        dataset.pop('_id')
        print(json.dumps(dataset, indent=2))

        break


    # run = runs[0]
    # run.pop('_id')
    # print(json.dumps(run, indent=2))
    # problem_ids = [run['problem']['id'] for run in runs]
    # print(len(problem_ids))
    # print(problem_ids)

    # problems = dc.get_problem_docs(query={'about.problemID': {'$in': problem_ids}})
    # print(problems.count())
    # problem = problems[0]
    # problem.pop('_id')
    # print(json.dumps(problem, indent=2))
    # for problem in problems:
    #     assert len(problem['inputs']['data']) == 1
    #     dataset_id = problem['inputs']['data'][0]['datasetID']
    #     assert len(problem['inputs']['data']['targets']) == 1
    #     target = problem['inputs']['data']['targets'][0]['colName']

    # dataset = dc.get_dataset_doc(query={'about.datasetID': '4550_MiceProtein_dataset'})
    # dataset.pop('_id')
    # print(json.dumps(dataset, indent=2))

    # primitives = set()
    # for pipe in pipes:
    #     steps = pipe['steps']
    #     for step in steps:
    #         primitives.add(step['primitive']['python_path'].replace('d3m.primitives.', ''))
    #         # print(json.dumps(step, indent=4))
    # print(len(primitives))
    # for p in sorted(primitives):
    #     print(p)


"""
--------------------
PRIMITIVES:
--------------------

Classifiers (14):
    classification.bagging.SKlearn
    classification.bernoulli_naive_bayes.SKlearn
    classification.decision_tree.SKlearn
    classification.extra_trees.SKlearn
    classification.gaussian_naive_bayes.SKlearn
    classification.gradient_boosting.SKlearn
    classification.k_neighbors.SKlearn
    classification.linear_discriminant_analysis.SKlearn
    classification.linear_svc.SKlearn
    classification.logistic_regression.SKlearn
    classification.passive_aggressive.SKlearn
    classification.random_forest.SKlearn
    classification.sgd.SKlearn
    classification.svc.SKlearn
Preprocessors (5):
    *data_preprocessing.EnsembleVoting.DSBOX*
    data_preprocessing.min_max_scaler.SKlearn
    data_preprocessing.nystroem.SKlearn
    data_preprocessing.random_sampling_imputer.BYU
    data_preprocessing.standard_scaler.SKlearn
Data Transformation (9):
    *data_transformation.column_parser.DataFrameCommon*
    *data_transformation.construct_predictions.DataFrameCommon*
    *data_transformation.dataset_to_dataframe.Common*
    *data_transformation.extract_columns_by_semantic_types.DataFrameCommon*
    data_transformation.fast_ica.SKlearn
    *data_transformation.horizontal_concat.DataFrameConcat*
    data_transformation.kernel_pca.SKlearn
    data_transformation.pca.SKlearn
    *data_transformation.rename_duplicate_name.DataFrameCommon*
Feature Selection (3):
    feature_selection.generic_univariate_select.SKlearn
    feature_selection.select_fwe.SKlearn
    feature_selection.select_percentile.SKlearn
Regression (17):
    regression.ard.SKlearn
    regression.decision_tree.SKlearn
    regression.extra_trees.SKlearn
    regression.gaussian_process.SKlearn
    regression.gradient_boosting.SKlearn
    regression.k_neighbors.SKlearn
    regression.kernel_ridge.SKlearn
    regression.lars.SKlearn
    regression.lasso.SKlearn
    regression.lasso_cv.SKlearn
    regression.linear_svr.SKlearn
    regression.mlp.SKlearn
    regression.passive_aggressive.SKlearn
    regression.random_forest.SKlearn
    regression.ridge.SKlearn
    regression.sgd.SKlearn
    regression.svr.SKlearn
"""

