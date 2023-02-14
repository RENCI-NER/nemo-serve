import numpy as np
import pandas as pd
from functools import partial
from src.utils.SAPElastic import SAPElastic


def open_numpy_pickle(file_path):
    arr = np.load(file_path)
    return arr


def open_csv(file_path):
    return pd.read_csv(file_path, header=0, dtype=str)


def get_id_type_dict(file_path):
    all = open_csv(file_path)
    result = {}
    for i, t in zip(all['id'],all['type']):
        result[i] = t.strip("['").strip("']")
    return result


def iter_files(np_file, name_id_file, id_type_file):
    np_arr = open_numpy_pickle(np_file)
    n_id_arr = open_csv(name_id_file)
    type_id_dict = get_id_type_dict(id_type_file)
    counter = 0
    for vector, curie in zip(np_arr, n_id_arr['ID']):
        curie = curie.strip("['").strip("']")
        es_object = {
          "id" : curie,
          "embedding": vector.tolist(),
          "name" : n_id_arr['Name'][counter],
          "category": type_id_dict[curie]
        }
        counter += 1
        yield es_object


def index_docs(elastic_connection, np_file, name_id_file, id_type_file):
    client = SAPElastic(**elastic_connection)
    client.delete_index()
    client.create_index()
    client.populate_index(partial(iter_files, np_file, name_id_file, id_type_file))