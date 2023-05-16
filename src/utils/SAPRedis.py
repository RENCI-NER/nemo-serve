import logging

import redis
import  redis.asyncio.client as AsyncRedis
from redis.commands.search.field import TextField, VectorField, TagField
from redis.commands.search.indexDefinition import IndexType , IndexDefinition
from redis.commands.search.query import Query
import numpy as np



class RedisMemory:
    DOC_PREFIX = "sap_docs:"
    SCHEMA = (
        TextField("name"),
        TextField("curie"),
        # @TODO if this preforms worse use categories as index prefix ?
        TagField("categories"),
        VectorField(
            "embedding",
            "HNSW",
            {"TYPE": "FLOAT32", "DIM": 768, "DISTANCE_METRIC": "COSINE"},
        ),
    )
    def __init__(self, host, port, password, index="sap_index"):
        self.host = host
        self.port = port
        self.password = password
        self.index_name = index
        self.sync_connection  = redis.Redis(
            host=self.host,
            port=self.port,
            password=self.password
        )
        try:
            self.sync_connection.ping()
            self.sync_connection.close()
        except redis.ConnectionError as con_error:
            logging.error(f"Error connection to redis {self.host}:{self.port} , {con_error}")
        self.async_connection = AsyncRedis.Redis(
            host=self.host,
            port=self.port,
            password=self.password
        )

    async def create_index(self):
        try:
            # check to see if index exists
            await self.async_connection.ft(self.index_name).info()
            logging.warning("Index already exists!")
        except:
            # index Definition
            definition = IndexDefinition(prefix=[RedisMemory.DOC_PREFIX], index_type=IndexType.HASH)

            # create Index
            await self.async_connection.ft(self.index_name).create_index(fields=RedisMemory.SCHEMA, definition=definition)

    async def delete_index(self):
        try:
            await self.async_connection.ft(self.index_name).dropindex(delete_documents=True)
        except Exception as e:
            logging.error(e)

    async def close(self):
        await self.async_connection.close()

    async def search(self, query_vector, top_n=10, bl_type="", algorithm=None):
        # search query
        base_query = f"*=>[KNN {top_n} @embedding $vector AS vector_score]"
        if bl_type:
            base_query = f"@categories: {bl_type}=>[KNN {top_n} @embedding $vector AS vector_score] "
        query = (
            # query
            Query(base_query)
                 # returned fields
                .return_fields("curie", "name", "categories", "vector_score")
                # sorting order
                .sort_by("vector_score")
                .dialect(2)
        )
        # convert query vector to bytes
        query_vector = np.array(query_vector).astype(np.float32).tobytes()

        try:
            # perform search query
            results = await self.async_connection.ft(f"{self.index_name}").search(
                query, query_params={"vector": query_vector}
            )
        except Exception as e:
            logging.warning("Error calling Redis search: ", e)
            # if error return None
            return None
        # return array of results with distance scores
        return [{"name": result.name,"curie": result.curie, "category": result.categories, "score": result.vector_score} for result in results.docs]

    async def populate_index(self, generator):
        # loop through the generator
        # pipeline chunk size
        chunk_size = 10_000
        counter = 1
        # aquire pipeline
        pipe = await self.async_connection.pipeline()
        for data in generator():
            vector = data['embedding']
            # convert vector to bytes
            vector = np.array(vector).astype(np.float32).tobytes()
            # create value for hash
            mapping = {
                "curie": data['curies'][0],
                "embedding": vector,
                "name": data['name'],
                "categories": data['categories'][0]
            }
            # redis hash key
            hash_key = f"{self.DOC_PREFIX}{mapping['curie'].replace(':', '_')}"
            await pipe.hset(hash_key, mapping=mapping)
            if counter % chunk_size == 0:
                # every 10K rounds execute pipeline
                await pipe.execute()
                logging.info(f"processed {counter} entries")
            counter += 1
        # execute any remaining statements on the pipeline
        await pipe.execute()

