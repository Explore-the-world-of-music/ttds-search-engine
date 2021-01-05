"""
This module runs all the neccessary steps
"""

# Todo: Integrate with frontend and backend. Currently main is used for testing only

import os
from ETL.preprocessing import Preprocessor
from search_engine.indexer import Indexer
from search_engine.retrieval import execute_queries_and_save_results
from helpers.misc import load_yaml, load_queries

if __name__ == "__main__":

    # Load config
    config = load_yaml("config/config.yaml")

    # Initialize preprocessor
    preprocessor = Preprocessor(config)

    # Load data
    doc_ids, raw_doc_texts = preprocessor.load_data("data/DUMMY_trec.5000.xml")

    # Initiate indexer
    indexer = Indexer(config)

    # Build index
    # indexer.build_index(preprocessor, doc_ids, raw_doc_texts)

    # Save index
    # indexer.store_index()

    # Add doc ids as index attribute
    indexer.add_all_doc_ids(doc_ids)

    # Load index (for testing)
    indexer.index = indexer.load_index()

    # Load boolean queries
    # Todo: Embed into final infrastructure - one query at a time?
    queries_num, queries = load_queries('queries.boolean.txt')

    # execute_queries_and_save_results
    # Todo: Discuss how we determine that a query is bool or not
    # Todo: Discuss if no class for the retrieval as search strategies are too different for one class
    # Todo: but too similar for different classes?
    for query_num, query in zip(queries_num, queries):
        execute_queries_and_save_results("Boolean_" + query_num + '_results_queries.txt', query_num, query,
                                         bool = True, indexer = indexer, preprocessor=preprocessor)

    # Load ranked queries
    # Todo: Embed into final infrastructure - one query at a time?
    queries_num, queries = load_queries('queries.ranked.txt')

    # execute_queries_and_save_results
    for query_num, query in zip(queries_num, queries):
        execute_queries_and_save_results("Ranked_" + query_num + '_results_queries.txt', query_num, query,
                                         bool = False, indexer = indexer, preprocessor=preprocessor)





