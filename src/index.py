import numpy as np
import pandas as pd
from functools import partial
import ast
import asyncio
from src.utils.SAPQdrant import SAPQdrant
from src.utils.SAPRedis import RedisMemory
import logging

logger = logging.getLogger()


def open_numpy_pickle(file_path, sub_path=""):
    print("subpath: ", sub_path)
    arr = np.load(file_path)
    if sub_path:
        return arr[sub_path]
    return arr


def open_csv(file_path):
    return pd.read_csv(file_path, header=0, dtype=str)


def get_id_type_dict(file_path):
    all = open_csv(file_path)
    result = {}
    for i, t in zip(all['id'],all['type']):
        # convert list string to actual list
        result[i] = ast.literal_eval(t)[0]
    return result


def iter_files(np_file, name_id_file, id_type_file, index_name, np_key=""):
    logger.info("opening np array file")
    np_arr = open_numpy_pickle(np_file, np_key)
    logger.info(f"found {np_arr.size} vector rows.")
    logger.info("opening name id csv file")
    n_id_arr = open_csv(name_id_file)
    logger.info("found {len(n_id_arr)} names id pairs.")
    type_id_dict = get_id_type_dict(id_type_file)
    logger.info("generating rows...")
    total_rows = len(n_id_arr)
    counter = 0
    for vector, curies in zip(np_arr, n_id_arr['ID']):
        # convert list string to actual list
        curies = ast.literal_eval(curies)
        doc = {
          "curies": curies,
          "embedding": vector,
          "name" : n_id_arr['Name'][counter],
          "categories": [ type_id_dict[c] for c in curies]
        }
        counter += 1
        if counter % 100_000 == 0:
            logger.info(f"generated {round(counter/total_rows, 2)* 100} %")
        yield doc



def index_docs(storage, connection_params, np_file, name_id_file, id_type_file, np_key=""):
    if storage=="qdrant":
        client = SAPQdrant(**connection_params)
    elif storage=="redis":
        client = RedisMemory(**connection_params)
    else:
        raise ValueError(f"Unsupported storage: {storage}")
    loop = asyncio.get_event_loop()
    loop.run_until_complete(client.delete_index())
    loop.run_until_complete(client.create_index())
    loop.run_until_complete(client.populate_index(partial(iter_files, np_file, name_id_file, id_type_file, connection_params['index'], np_key)))
    loop.run_until_complete(client.refresh_index())
    loop.run_until_complete(client.close())

