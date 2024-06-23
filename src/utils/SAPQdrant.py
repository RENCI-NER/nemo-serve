import logging
from qdrant_client import AsyncQdrantClient, models
import numpy as np

logger = logging.getLogger()


class SAPQdrant:
    def __init__(self, host, index, default_timeout=1000, max_retries=10, retry_on_timeout=True
                 , vector_similarity="dot_product", scheme="https", port="443", *args, **kwargs):
        self.client = AsyncQdrantClient(
            url=f"{scheme}://{host}:{port}",
        )
        self.index = index

    async def delete_index(self):
        exists = await self.client.collection_exists(collection_name=self.index)
        if exists:
            return await self.client.delete_collection(collection_name=self.index)

    async def create_index(self):
        logger.info('Creating index')
        return await self.client.create_collection(
            collection_name=self.index,
            vectors_config=models.VectorParams(
                size=768,
                distance="Cosine"
            ),
            # hswn_config=models.HnswConfig(
            #     m=16,
            #     ef_construct=100
            # ),

        )

    async def disable_indexing(self):
        await self.client.update_collection(
            collection_name=self.index,
            optimizer_config=models.OptimizersConfigDiff(
                indexing_threshold=0
            )
        )

    async def enable_indexing(self):
        await self.client.update_collection(
            collection_name=self.index,
            optimizer_config=models.OptimizersConfigDiff(
                indexing_threshold=20_000
            )
        )

    async def populate_index(self, generator, counter=0):
        # turn of internal indexing for speed.
        # https://qdrant.tech/documentation/tutorials/bulk-upload/#disable-indexing-during-upload
        await self.disable_indexing()
        chunk_size = 1_000
        to_insert = []
        for data in generator():
            vector = data['embedding']
            # convert vector to bytes
            vector = np.array(vector).astype(np.float32)  # .tobytes()
            payload = {
                "curie": data['curies'][0],
                "name": data['name'],
                "category": data['categories'][0]
            }
            to_insert += [
                models.PointStruct(id=counter, vector=vector, payload=payload)
            ]

            if counter % chunk_size == 0:
                operation_info = await self.client.upsert(
                    collection_name=self.index,
                    wait=True,
                    points=to_insert
                )
                logger.debug(f'Inserted {len(to_insert)}; {operation_info}')
                to_insert = []
            counter += 1
        # insert remaining
        if len(to_insert):
            operation_info = await self.client.upsert(
                collection_name=self.index,
                wait=True,
                points=to_insert
            )
            logger.info(f'Inserting remaining {len(to_insert)}; {operation_info}')
            # revert indexing scheme
        await self.enable_indexing()


    async def search(self, query_vector, top_n=10, bl_type=None, *args, **kwargs):
        if bl_type:
            results = await self.client.search(
                collection_name=self.index,
                query_vector=query_vector,
                with_payload=True,
                limit=top_n,
                query_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="categories",
                            match=models.MatchValue(
                                value=bl_type
                            )
                        )
                    ]
                )
            )
        else:
            results = await self.client.search(
                collection_name=self.index,
                query_vector=query_vector,
                with_payload=True,
                limit=top_n,
            )
        return [
            {
                "score": x.score,
                # @TODO when loading rename this field
                "category": x.payload["categories"],
                "name": x.payload["name"],
                "curie": x.payload["curie"]
            } for x in results
        ]

    async def refresh_index(self):
        await self.client.create_payload_index(
            collection_name=f"{self.index}",
            field_name="category",
            field_schema="keyword",
        )
