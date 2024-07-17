import os
import json
import pandas as pd
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from eland.ml.ltr import LTRModelConfig, QueryFeatureExtractor, FeatureLogger
import xgboost as xgb
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Load environment variables from .env file
load_dotenv()

# Elasticsearch configuration
ELASTICSEARCH_HOST = os.getenv("ELASTICSEARCH_HOST", "localhost")
ELASTICSEARCH_PORT = int(os.getenv("ELASTICSEARCH_PORT", "9200"))
ELASTICSEARCH_USERNAME = os.getenv("ELASTICSEARCH_USERNAME")
ELASTICSEARCH_PASSWORD = os.getenv("ELASTICSEARCH_PASSWORD")

# Initialize Elasticsearch client
es_client = Elasticsearch(
    hosts=[{"host": ELASTICSEARCH_HOST, "port": ELASTICSEARCH_PORT, "scheme": "http"}],
    basic_auth=(ELASTICSEARCH_USERNAME, ELASTICSEARCH_PASSWORD)
)

index_name = "anime"

# Define LTR model configuration
ltr_config = LTRModelConfig(
    feature_extractors=[
        QueryFeatureExtractor(feature_name="title_bm25", query={"match": {"title": "{{query}}"}}),
        QueryFeatureExtractor(feature_name="synopsis_bm25", query={"match": {"synopsis": "{{query}}"}}),
        QueryFeatureExtractor(feature_name="phrase_match_title", query={"match_phrase": {"title": "{{query}}"}}),
        QueryFeatureExtractor(feature_name="phrase_match_synopsis", query={"match_phrase": {"synopsis": "{{query}}"}}),
        QueryFeatureExtractor(feature_name="match_synopsis_and", query={"match": {"title": {"query": "{{query}}", "operator": "and"}}}),
        QueryFeatureExtractor(feature_name="bm25_genre", query={"match": {"genres": "{{query}}"}}),
        QueryFeatureExtractor(
            feature_name="bm25_studios",
            query={"match": {"studios": "{{query}}"}}
        )
    ]
)

# Initialize FeatureLogger
feature_logger = FeatureLogger(es_client, index_name, ltr_config)

def to_named_query(query, query_name):
    return {"bool": {"must": query, "_name": query_name}}

def extract_query_features_without_docs(query_params):
    from elasticsearch._sync.client import _quote
    __path = f"/{_quote(index_name)}/_search/template"
    __query = {"include_named_queries_score": True}
    __headers = {"accept": "application/json", "content-type": "application/json"}

    query_extractors = feature_logger._model_config.query_feature_extractors
    queries = [to_named_query(extractor.query, extractor.feature_name) for extractor in query_extractors]

    fields_to_get = [
        "anime_id", "anime_url", "title", "synopsis", "main_pic", "type", 
        "source_type", "num_episodes", "status", "start_date", "end_date", 
        "season", "studios", "genres", "score", "pics"
    ]
    additional_features = ["score", "score_count", "score_rank"]
    feat_names = [extractor.feature_name for extractor in query_extractors]

    source = json.dumps({
        "query": {"bool": {"should": queries}},
        "_source": fields_to_get + additional_features,
        "size": 100
    })

    __body = {"source": source, "params": {**query_params}}

    complete_response =  es_client.perform_request("GET", __path, params=__query, headers=__headers, body=__body)
    response = complete_response["hits"]["hits"]

    documents = {}
    animes_response = {}

    for hit in response:
        doc_id = hit['_source']['anime_id']
        dic_features = {
            feature: hit['matched_queries'][feature] if 'matched_queries' in hit and feature in hit['matched_queries'] else 0
            for feature in feat_names
        }
        animes_response[doc_id] = {field: hit['_source'][field] for field in fields_to_get}
        for feature in additional_features:
            dic_features[feature] = hit['_source'][feature]

        documents[doc_id] = dic_features

    df = pd.DataFrame.from_dict(documents, orient='index').reset_index().rename(columns={"index": "anime_id"})
    df2 = pd.DataFrame.from_dict(animes_response, orient='index').reset_index().rename(columns={"index": "anime_id"})
    
    return df, df2, int(complete_response['took'])

# Load the XGBoost model
model_xgb = xgb.Booster()
model_xgb.load_model('anime_search_sdbn.txt')

@app.route('/search_ltr', methods=['POST'])
def search_ltr():
    try:
        data = request.get_json()
        query = data.get('query')
        
        if not query:
            return jsonify({"error": "Query is required"}), 400
        
        from_param = request.args.get('from', default=0, type=int)
        size_param = request.args.get('size', default=10, type=int)

        
        # Extract features and anime details
        features, animes, took = extract_query_features_without_docs({"query": query})

        feature_names_to_predict = features.columns.drop('anime_id')

        # Predict relevance scores
        dmatrix = xgb.DMatrix(features[feature_names_to_predict])
        y_pred = model_xgb.predict(dmatrix)

        # Sort animes by relevance
        animes['relevance'] = y_pred
        animes = animes.sort_values(by='relevance', ascending=False)

        # Drop the 'relevance' column
        animes = animes.drop(columns=['relevance'])

        # Ensure columns are unique (if needed)
        animes = animes.loc[:, ~animes.columns.duplicated()]

        # Convert to JSON format
        animes_json = animes.to_dict(orient='records')

        animes_json_from = animes_json[from_param:]
        
        animes_json_size = animes_json_from[:size_param]
        
        return jsonify({ "animes": animes_json_size, "took": took})
    except Exception as e:
        print(e)
        return jsonify({ "animes": [], "took": 0})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
