"""
This module runs all the neccessary steps
"""

# Todo: Integrate with frontend and backend. Currently main is used for testing only

from ETL.preprocessing import Preprocessor
from helpers.misc import load_yaml, load_queries
from search_engine.indexer import Indexer
from search_engine.retrieval import execute_queries_and_save_results
from datetime import datetime
import logging

if __name__ == "__main__":

    # Stop time
    # Full run: 23 seconds
    # Run without index: 0-1 seconds
    dt_string_START = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    logging.warning(f'START date and time = {dt_string_START}')

    # Load config
    config = load_yaml("config/config.yaml")

    # Initialize preprocessor instance
    preprocessor = Preprocessor(config)

    # Load data
    doc_ids, raw_doc_texts = preprocessor.load_data("data/DUMMY_trec.5000.xml")

    # Initiate indexer instance
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
    # # Todo: Embed into final infrastructure
    # queries_num, queries = load_queries('queries.boolean.txt')
    #
    # # Execute Boolean queries for CW1
    # # Todo: Discuss how we determine that a query is bool or not
    # # Todo: Discuss if no class for the retrieval as search strategies are too different for one class
    # # Todo: but too similar for different classes?
    # results = ""
    # for query_num, query in zip(queries_num, queries):
    #     results_tmp = execute_queries_and_save_results(query_num, query, search_type="boolean", indexer=indexer,
    #                                                 preprocessor=preprocessor, config=config)
    #     if results_tmp is not None:
    #         results = results + results_tmp
    #
    # with open("data/results/Boolean_" + '_results_queries.txt', mode="w", encoding="utf-8") as f:
    #     f.writelines(results[:-1])
    #
    #
    # Load ranked queries
    # queries_num, queries = load_queries('queries.ranked.txt')
    #
    # # Execute Ranked queries for CW1
    # results = ""
    # for query_num, query in zip(queries_num, queries):
    #     results_tmp = execute_queries_and_save_results(query_num, query, search_type="tfidf", indexer=indexer,
    #                                                preprocessor=preprocessor, config=config)
    #     if results_tmp is not None:
    #         results = results + results_tmp
    # with open("data/results/Ranked_" + '_results_queries.txt', mode="w", encoding="utf-8") as f:
    #     f.writelines(results[:-1])

    # Load boolean + ranked queries
    queries_num, queries = load_queries('queries.boolean_and_ranked.txt')

    # Execute Boolean Ranked queries
    results = ""
    for query_num, query in zip(queries_num, queries):
        results_tmp = execute_queries_and_save_results(query_num, query, search_type="boolean_and_tfidf", indexer=indexer,
                                                       preprocessor=preprocessor, config=config)
        if results_tmp is not None:
            results = results + results_tmp
    with open("data/results/Ranked_AND_Boolean" + '_results_queries.txt', mode="w", encoding="utf-8") as f:
        f.writelines(results[:-1])

    dt_string_END = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    logging.warning(f'END date and time = {dt_string_END}')
