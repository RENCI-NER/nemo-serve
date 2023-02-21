import json
import logging


from elasticsearch import AsyncElasticsearch, helpers
# from elasticsearc
from enum import Enum, EnumMeta
import numpy as np

class MetaEnum(EnumMeta):
    def __contains__(cls, item):
        try:
            cls(item)
        except ValueError:
            return False
        return True
class BaseEnum(Enum, metaclass=MetaEnum):
    pass
class DENSE_VECTOR_SIMILARITY(BaseEnum):
    DOT_PRODUCT = "dot_product"
    L2_SIMILARITY = "l2_norm"
    COSINE = "cosine"

class SAPElastic:

    def __init__(self, host, username, password, index, default_timeout=1000, max_retries=10, retry_on_timeout=True
                 , vector_similarity=DENSE_VECTOR_SIMILARITY.DOT_PRODUCT):
        assert vector_similarity in DENSE_VECTOR_SIMILARITY, f"Vector similarity needs to be one of dot_product, cosine or l2_norm." \
                                                             f"https://www.elastic.co/guide/en/elasticsearch/reference/current/dense-vector.html#dense-vector-params"
        self.dense_vector_similarity = vector_similarity.value
        self.es_client = AsyncElasticsearch(
            [host],
            basic_auth=(username, password if password else ""),
            request_timeout=default_timeout,
            max_retries=max_retries,
            retry_on_timeout=retry_on_timeout
        )
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
        self.es_client.indices.put_settings(index=index_name,
                                            settings={
                                                "index.refresh_interval": "300s"
                                            })

    async def close(self):
        await self.es_client.close()

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
                    "similarity": self.dense_vector_similarity
                },
                "name": {
                    "type": "text"
                },
                "categories": {
                    "type": "keyword",
                    "store": True
                },
                "curies": {
                    "type": "text"
                }
            }
        }

    async def populate_index(self, generator):
        normalize = False
        if self.dense_vector_similarity == DENSE_VECTOR_SIMILARITY.DOT_PRODUCT.value:
            normalize = True
        await helpers.async_bulk(self.es_client, generator(normalize=normalize), chunk_size=1_000_000, max_retries=3, initial_backoff=2)
        self.es_client.indices.refresh(index=self.index)

    async def delete_index(self):
        await self.es_client.options(ignore_status=[400,404]).indices.delete(index=self.index)

    async def get_docs_count(self):
        return await self.es_client.count(index=self.index)

    async def search_knn(self, query_vector, top_n=3, bl_type=""):
        search_query = {
            "knn": {
                "field": "embedding",
                "query_vector": query_vector,
                "k": top_n,
                # defaulting to consider 10000 docs as candidates,
                "num_candidates": 10000
            },
            "source": [
                "curies",
                "name",
                "categories"
            ]
        }
        filter = None
        if bl_type:
            search_query["knn"]["filter"] = {"term": {"categories": bl_type}}
        return await self.es_client.search(index=self.index,
                                     knn=search_query['knn'],
                                     source =search_query['source'],
                                     size=top_n
                                         )

    async def search_cosine(self, query_vector, top_n=10, bl_type=""):
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
                "curies",
                "name",
                "categories"
            ]
        }
        return await self.es_client.search(index=self.index,
                                     query=search_query["query"],
                                     size=top_n,
                                     source=search_query["source"])

    async def search(self, query_vector, top_n=10, bl_type="", algorithm="cosine"):

        if self.dense_vector_similarity == DENSE_VECTOR_SIMILARITY.DOT_PRODUCT.value:
            query_vector = self.normalize_vector(query_vector)

        if algorithm == "cosine":
            es_response = await self.search_cosine(query_vector=query_vector,
                                      top_n=top_n,
                                      bl_type=bl_type)
        elif algorithm == "knn":
            es_response = await self.search_knn(query_vector=query_vector,
                                      top_n=top_n,
                                      bl_type=bl_type)
        result = []
        for items in es_response.body['hits']['hits']:
            attrs = items['_source']
            score = items['_score']
            name = attrs['name']
            for curie, bl_type in zip(attrs['curies'], attrs['categories']):
                result.append({
                    "label": name,
                    "curie": curie,
                    "category": bl_type,
                    "score": score,
                    })
        result.sort(
            key=lambda x: x['score'],
            reverse=True
        )
        return  {
            "performance_meta": {
                "took": es_response.body['took'],
                "timed_out": es_response.body['timed_out']
            },
            "results" : result
        }


    @staticmethod
    def normalize_vector(v):
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm


if __name__ == "__main__":
    # not using password huh
    index_name = "test_index"
    import asyncio
    es_client = SAPElastic("http://localhost:9200", "elastic", password=None, index=index_name, vector_similarity=DENSE_VECTOR_SIMILARITY.DOT_PRODUCT)
    with open('../sample.json') as stream:
        items = json.load(stream)

    def generate_items(normalize=False):
        if normalize:
            formatted_items = [
                {
                    'curies': [i['id'], i['id']],
                    'embedding': SAPElastic.normalize_vector(i['embedding']),
                    'categories': [i['category'], i['category']],
                    'name': i['name']
                } for i in items
            ]
        else:
            formatted_items = [
                {
                    'curies': [i['id'], i['id']],
                    'embedding': SAPElastic.normalize_vector(i['embedding']),
                    'categories': [i['category'], i['category']],
                    'name': i['name']
                } for i in items
            ]
        return [
            {'_index': index_name,
             '_source': x} for x in formatted_items
        ]
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(es_client.search(query_vector=items[1]['embedding'], top_n=10, algorithm="knn", bl_type=""))
    loop.run_until_complete(es_client.close())
    for h in result['results']:
        print(h['curie'], h['label'], h['category'], h['score'])
    # #
    # # print('------------------')
    # result = es_client.search(query_vector=items[1]['embedding'], top_n=10, algorithm="cosine")
    # for h in result.body['hits']['hits']:
    #     score = h['_score']
    #     h = h['_source']
    #     print(h['curies'], h['name'], h['categories'], score)
    # es_client.delete_index()
    # es_client.create_index()
    # es_client.populate_index(generate_items)
