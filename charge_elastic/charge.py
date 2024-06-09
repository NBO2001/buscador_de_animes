import csv
import os
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from pprint import pprint

# Load environment variables from .env file
load_dotenv()

ELASTICSEARCH_HOST = os.getenv("ELASTICSEARCH_HOST", "localhost")
ELASTICSEARCH_PORT = int(os.getenv("ELASTICSEARCH_PORT", "9200"))
ELASTICSEARCH_USERNAME = os.getenv("ELASTICSEARCH_USERNAME")
ELASTICSEARCH_PASSWORD = os.getenv("ELASTICSEARCH_PASSWORD")

class Search:
    def __init__(self):
        # Initialize Elasticsearch connection with authentication
        self.es = Elasticsearch(
            hosts=[{
                "host": ELASTICSEARCH_HOST,
                "port": ELASTICSEARCH_PORT,
                "scheme": "http"
            }],
            basic_auth=(ELASTICSEARCH_USERNAME, ELASTICSEARCH_PASSWORD)
        )
        
        # Check if the connection is successful
        try:
            client_info = self.es.info()
            print('Connected to Elasticsearch!')
            pprint(client_info)
        except Exception as e:
            print(f"Error connecting to Elasticsearch: {e}")

    def create_index(self, index_name, settings):
        # Create an index with the given settings
        try:
            self.es.indices.create(index=index_name, body=settings)
            print(f"Index '{index_name}' created successfully.")
        except Exception as e:
            print(f"Error creating index '{index_name}': {e}")

    def index_data(self, index_name, data):
        # Index the data into Elasticsearch
        for i, record in enumerate(data):
            try:
                self.es.index(index=index_name, id=i+1, document=record)
            except Exception as e:
                print(f"Error indexing record {i+1}: {e}")
                print(record)
                exit()

# Create an instance of the Search class to test the connection
search_instance = Search()

# Define the index settings and mappings
index_name = "anime"
settings = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "analysis": {
            "analyzer": {
                "gen_analyzer": {
                "tokenizer": "keyword",
                "filter": [ "word_delimiter", "lowercase" ]
                }
            }
        }
    },
    "mappings": {
        "properties": {
            "anime_id": {"type": "integer"},
            "anime_url": {"type": "keyword"},
            "title": {"type": "text"},
            "synopsis": {"type": "text"},
            "main_pic": {"type": "keyword"},
            "type": {"type": "keyword"},
            "source_type": {"type": "keyword"},
            "num_episodes": {
                "type": "rank_feature"
            },
            "status": {"type": "keyword"},
            "start_date": {"type": "date"},
            "end_date": {"type": "date"},
            "season": {"type": "keyword"},
            "studios": {"type": "keyword"},
            "genres": {
                "type": "text",  
                "analyzer": "gen_analyzer" 
            },
            "score": {"type": "rank_feature"},
            "score_count": {"type": "rank_feature"},
            "score_rank": {"type": "integer"},
            "popularity_rank": {"type": "integer"},
            "members_count": {"type": "integer"},
            "favorites_count": {"type": "integer"},
            "watching_count": {"type": "integer"},
            "completed_count": {"type": "rank_feature"},
            "on_hold_count": {"type": "integer"},
            "dropped_count": {
                "type": "rank_feature",
                "positive_score_impact": False
            },
            "plan_to_watch_count": {"type": "integer"},
            "total_count": {"type": "integer"},
            "score_10_count": {"type": "integer"},
            "score_09_count": {"type": "integer"},
            "score_08_count": {"type": "integer"},
            "score_07_count": {"type": "integer"},
            "score_06_count": {"type": "integer"},
            "score_05_count": {"type": "integer"},
            "score_04_count": {"type": "integer"},
            "score_03_count": {"type": "integer"},
            "score_02_count": {"type": "integer"},
            "score_01_count": {"type": "integer"},
            "clubs": {"type": "keyword"},
            "pics": {"type": "keyword"}
        }
    }
}


if not search_instance.es.indices.exists(index=index_name):
    search_instance.create_index(index_name, settings)

#Read the CSV file and prepare the data

data = []
csv_file_path = "anime.csv"

fisrt = True
with open(csv_file_path, mode='r', encoding='utf-8') as file:
    csv_reader = csv.DictReader(file, delimiter='\t')
    for row in csv_reader:
        
        try:
            for field in ['anime_id', 'num_episodes', 'popularity_rank', 'members_count', 'favorites_count', 'watching_count',
                          'completed_count', 'on_hold_count', 'dropped_count', 'plan_to_watch_count', 'total_count',
                          'score_10_count', 'score_09_count', 'score_08_count', 'score_07_count', 'score_06_count',
                          'score_05_count', 'score_04_count', 'score_03_count', 'score_02_count', 'score_01_count', 'score_count', 'score_rank']:
               
                if row[field]:
                    row[field] = int(row[field])
                else:
                    row[field] = 0

            for field_rank in ['dropped_count', 'score_count', 'completed_count', 'num_episodes']:
                if row[field_rank]:
                    row[field_rank] = int(row[field_rank])
                    row[field_rank] = row[field_rank] if row[field_rank] > 0 else 1
                else:
                    row[field_rank] = 1

            if row['score']:
                row['score'] = float(row['score'])
                row['score'] = row['score'] if row['score'] > 0 else 0.1
            else:
                row['score'] = 0.1
                    
            # Parse start_date
            if row['start_date']:
                row['start_date'] = "T".join(row['start_date'].split(" "))
            else:
                row['start_date'] = None
            
            # Parse end_date
            if row['end_date']:
                row['end_date'] = "T".join(row['end_date'].split(" "))
            else:
                row['end_date'] = None
            
            data.append(row)
            
        except Exception as e:
            print(f"Error processing row: {e}")
            print(row['start_date'], row['end_date'])
            exit()

# Index the data into Elasticsearch
search_instance.index_data(index_name, data)
