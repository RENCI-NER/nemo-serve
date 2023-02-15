import json
import logging


from elasticsearch import Elasticsearch, helpers
from functools import partial

class SAPElastic:

    def __init__(self, host, username, password, index, default_timeout=1000, max_retries=10, retry_on_timeout=True):
        self.es_client = Elasticsearch(
            [host],
            basic_auth=(username, password if password else ""),
            timeout=default_timeout,
            max_retries=max_retries,
            retry_on_timeout=retry_on_timeout
        )
        resp = self.es_client.ping()
        if resp:
            logging.info(f"Pinging Elastic success... connected to {host}")
        else:
            logging.error(f"Failed to ping Elastic @ {host}")
            raise Exception("Cannot ping Elastic...")
        self.index = index

    def create_index(self):
        """
        Create an ES index.
        :param index_name: Name of the index.
        :param mapping: Mapping of the index
        """
        mapping = self._get_mapping()
        index_name = self.index
        logging.info(f"Creating index {index_name} with the following schema: {json.dumps(mapping, indent=2)}")

        self.es_client.options().indices.create(index=index_name, mappings=mapping)


    def _get_mapping(self):
        """
        Returns elastic index definition
        :return:
        """
        return {
            "properties": {
              "embedding": {
                "type": "dense_vector",
                "dims": 768,
                "index": True,
                # https://www.elastic.co/guide/en/elasticsearch/reference/current/dense-vector.html#dense-vector-similarity
                "similarity": "l2_norm"
              },
              "name": {
                "type": "text"
              },
              "category": {
                "type": "keyword"
              },
              "id": {
                 "type": "text"
              }
            }
        }

    def generate_docs(self, file_path):
        with open(file_path) as stream:
            items = json.load(stream)
        for l, i in enumerate(items):
            yield {
                "_index": self.index,
                "_source": i
            }

    def populate_index(self, generator):
        helpers.bulk(self.es_client, generator())

    def delete_index(self):
        self.es_client.options(ignore_status=[400,404]).indices.delete(index=self.index)

    def get_docs_count(self):
        return self.es_client.count(index=self.index)

    def search_knn(self, query_vector, top_n=3, bl_type=""):
        search_query = {
            "knn": {
                "field": "embedding",
                "query_vector": query_vector,
                "k": top_n,
                # defaulting to consider 10000 docs as candidates,
                "num_candidates": 10000
            },
            "source": [
                "id",
                "name",
                "category"
            ]
        }
        filter = None
        if bl_type:
            filter = {"term": {"category": bl_type}}
        return self.es_client.knn_search(index=self.index, knn=search_query['knn'], source =search_query['source']
                                         , filter=filter)

    def search_cosine(self, query_vector, top_n=10, bl_type=""):
        match_condition = {"match_all": {}}
        if bl_type:
            match_condition = {"bool" : {"filter" : {"term" : {"category" : bl_type}}}}
        search_query = {
            "query": {
                "script_score": {
                  "query" : match_condition,
                  "script": {
                    "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                    "params": {
                      "query_vector": query_vector
                    }
                  }
                }
            },
            "source": [
                "id",
                "name",
                "category"
            ]
        }
        return self.es_client.search(index=self.index,
                                     query=search_query["query"],
                                     size=top_n,
                                     source=search_query["source"])



if __name__ == "__main__":
    # not using password huh
    index_name = "sap_index"
    es_client = SAPElastic("http://localhost:9900", "elastic", password=None, index=index_name)
    with open('../sample.json') as stream:
        items = json.load(stream)
    # print(len())
    # result = es_client.search_cosine(query_vector=items[1]['embedding'], top_n=10)
    # for h in result['hits']['hits']:
    #     score = h['_score']
    #     h = h['_source']
    #     print(h['id'], h['name'], h['category'], score)
    #
    # print('------------------')
    # result = es_client.search_knn(query_vector=items[1]['embedding'], top_n=10)
    # for h in result['hits']['hits']:
    #     score = h['_score']
    #     h = h['_source']
    #     print(h['id'], h['name'], h['category'], score)
    es_client.delete_index()
    es_client.create_index()
    es_client.populate_index(partial(es_client.generate_docs, "../sample.json"))
