import os
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
import pandas as pd
import numpy as np
from eland.ml.ltr import LTRModelConfig, QueryFeatureExtractor, FeatureLogger
from xgboost import XGBRanker
from sklearn.model_selection import GroupShuffleSplit

# Load environment variables from .env file
load_dotenv()

ELASTICSEARCH_HOST = os.getenv("ELASTICSEARCH_HOST", "localhost")
ELASTICSEARCH_PORT = int(os.getenv("ELASTICSEARCH_PORT", "9200"))
ELASTICSEARCH_USERNAME = os.getenv("ELASTICSEARCH_USERNAME")
ELASTICSEARCH_PASSWORD = os.getenv("ELASTICSEARCH_PASSWORD")

es_client = Elasticsearch(
    hosts=[{
        "host": ELASTICSEARCH_HOST,
        "port": ELASTICSEARCH_PORT,
        "scheme": "http"
    }],
    basic_auth=(ELASTICSEARCH_USERNAME, ELASTICSEARCH_PASSWORD)
)

judgments_df = pd.read_csv("reference_colletion.csv")

index_name = "anime"

# Define LTR model configuration
ltr_config = LTRModelConfig(
    feature_extractors=[
        QueryFeatureExtractor(
            feature_name="title_bm25", 
            query={"match": {"title": "{{query}}"}}
        ),
        QueryFeatureExtractor(
            feature_name="synopsis_bm25", 
            query={"match": {"synopsis": "{{query}}"}}
        ),
        QueryFeatureExtractor(
            feature_name="phrase_match_title",
            query={"match_phrase": {"title": "{{query}}"}}
        ),
        QueryFeatureExtractor(
            feature_name="phrase_match_synopsis",
            query={"match_phrase": {"synopsis": "{{query}}"}}
        ),
        QueryFeatureExtractor(
            feature_name="match_synopsis_and",
            query={"match": {"title": {"query": "{{query}}", "operator": "and"}}}
        ),
        QueryFeatureExtractor(
            feature_name="bm25_genre",
            query={"match": {"genres": "{{query}}"}}
        ),
        QueryFeatureExtractor(
            feature_name="bm25_studios",
            query={"match": {"studios": "{{query}}"}}
        ),
        QueryFeatureExtractor(
            feature_name="score_pop",
            query={
                "script_score": {
                    "query": {"exists": {"field": "members_count"}},
                    "script": {"source": "return doc['members_count'].value/10000000.0"},
                }
            }
        )
    ]
)

# Initialize FeatureLogger
feature_logger = FeatureLogger(es_client, index_name, ltr_config)

import json

def to_named_query(query, query_name):
    return {"bool": {"must": query, "_name": query_name}}

def extract_query_features(query_params, doc_ids):
    __path = f"/anime/_search/template"
    __query = {"include_named_queries_score": True}
    __headers = {"accept": "application/json", "content-type": "application/json"}

    query_extractors = feature_logger._model_config.query_feature_extractors

    queries = [
        to_named_query(extractor.query, extractor.feature_name)
        for extractor in query_extractors
    ]

    feats_names = [extractor.feature_name for extractor in query_extractors]
    adicional_features = ["score", "score_count", "score_rank"]

    source = json.dumps({
        "query": {
            "bool": {
                "should": queries,
                "filter": [
                    {
                        "terms": {
                            "anime_id": doc_ids
                        }
                    }
                ]
            }
        },
        "_source": ["anime_id", "title"] + adicional_features
    })

    __body = {
        "source": source,
        "params": {**query_params},
    }

    response = es_client.perform_request("GET", __path, params=__query, headers=__headers, body=__body)["hits"]["hits"]

    documents = {}

    for hit in response:
        dic_features = {
            feature: hit['matched_queries'][feature] 
            if 'matched_queries' in hit and feature in hit['matched_queries'] 
            else 0 
            for feature in feats_names
        }

        doc_id = hit['_source']['anime_id']

        for new_feature in adicional_features:
            dic_features[new_feature] = hit['_source'][new_feature]

        documents[doc_id] = dic_features

    return documents, feats_names + adicional_features

def extract_query_features_without_docs(query_params):
    __path = f"/anime/_search/template"
    __query = {"include_named_queries_score": True}
    __headers = {"accept": "application/json", "content-type": "application/json"}

    query_extractors = feature_logger._model_config.query_feature_extractors

    queries = [
        to_named_query(extractor.query, extractor.feature_name)
        for extractor in query_extractors
    ]

    fields_to_get = [
        "anime_id", "anime_url", "title", "synopsis", "main_pic", "type", 
        "source_type", "num_episodes", "status", "start_date", "end_date", 
        "season", "studios", "genres", "score", "pics"
    ]
    feats_names = [extractor.feature_name for extractor in query_extractors]
    adicional_features = ["score", "score_count", "score_rank"]

    source = json.dumps({
        "query": {
            "bool": {
                "should": queries
            }
        },
        "_source": fields_to_get + adicional_features
    })

    __body = {
        "source": source,
        "params": {**query_params},
    }

    response = es_client.perform_request("GET", __path, params=__query, headers=__headers, body=__body)["hits"]["hits"]

    documents = {}
    animes_response = {}

    for hit in response:
        dic_features = {
            feature: hit['matched_queries'][feature] 
            if 'matched_queries' in hit and feature in hit['matched_queries'] 
            else 0 
            for feature in feats_names
        }

        doc_id = hit['_source']['anime_id']

        animes_response[doc_id] = {
            fet: hit['_source'][fet] for fet in fields_to_get
        }

        for new_feature in adicional_features:
            dic_features[new_feature] = hit['_source'][new_feature]

        documents[doc_id] = dic_features

        doc_ids = documents.keys()

        df = pd.DataFrame()
        df2 = pd.DataFrame()
        df['anime_id'] = doc_ids

        for feature_name in feats_names + adicional_features:
            df[feature_name] = np.array(
                [documents[doc_id][feature_name] for doc_id in doc_ids]
            )

        for feature_name in fields_to_get:
            df2[feature_name] = np.array(
                [animes_response[doc_id][feature_name] for doc_id in doc_ids]
            )

    return df, df2

def _extract_query_features(query_judgements_group):
    # Retrieve document ids in the query group as strings.
    doc_ids = query_judgements_group["document"].astype(int).tolist()

    # Resolve query params for the current query group
    query_params = {"query": query_judgements_group["query"].iloc[0]}

    try:
        # Extract the features for the documents in the query group:
        doc_features, fet_name = extract_query_features(query_params, doc_ids)
    except Exception as ke:
        print(f"Error extracting features for query: {query_params}, error: {ke}")
        # Handle the error as needed
        return None  # Or handle in a different way based on your application

    for feature_name in fet_name:
        query_judgements_group[feature_name] = np.array(
            [doc_features[doc_id][feature_name] if doc_id in doc_features.keys() else 0 for doc_id in doc_ids]
        )

    return query_judgements_group

# Assuming judgments_df is your DataFrame
judgments_with_features = judgments_df.groupby("query", group_keys=False).apply(_extract_query_features)
judgments_with_features = judgments_with_features.reset_index(drop=True)

features_names = list(judgments_with_features.columns)[3:]

# Shaping training and eval data in the expected format.
X = judgments_with_features[features_names]
y = judgments_with_features["label"]

groups = judgments_with_features["query"]

# Split the dataset in two parts respectively used for training and evaluation of the model.
group_preserving_splitter = GroupShuffleSplit(n_splits=1, train_size=0.7).split(
    X, y, groups
)
train_idx, eval_idx = next(group_preserving_splitter)

train_features, eval_features = X.loc[train_idx], X.loc[eval_idx]
train_target, eval_target = y.loc[train_idx], y.loc[eval_idx]
train_query_groups, eval_query_groups = groups.loc[train_idx], groups.loc[eval_idx]

# Create the ranker model:
ranker = XGBRanker(
    objective="rank:ndcg",
    eval_metric=["ndcg@10"],
    early_stopping_rounds=20
)

# Training the model
ranker.fit(
    X=train_features,
    y=train_target,
    group=train_query_groups.value_counts().sort_index().values,
    eval_set=[(eval_features, eval_target)],
    eval_group=[eval_query_groups.value_counts().sort_index().values],
    verbose=True,
)

ranker.save_model("../ltr_service/anime_search_sdbn.txt")
ranker.save_model("anime_search_sdbn.txt")
