"""
This module runs all the necessary steps to test functions
"""

# Todo: Integrate with frontend and backend. Currently main is used for testing only

import logging
from datetime import datetime
import pandas as pd

from ETL.preprocessing import Preprocessor
from helpers.misc import load_yaml, load_queries
from search_engine.indexer import Indexer
from search_engine.retrieval import execute_queries_and_save_results
from search_engine.system_evaluation import *

if __name__ == "__main__":

    # Stop time
    # Full run: 23 seconds
    # Run without creating but only loading index: 0-1 seconds
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
    indexer.build_index(preprocessor, doc_ids, raw_doc_texts)

    # Save index
    indexer.store_index(as_pickle=False)

    # Add doc ids as index attribute
    indexer.add_all_doc_ids(doc_ids)

    # Load index (for testing)
    indexer.index = indexer.load_index(as_pickle=False)

    # Load boolean queries
    queries_num, queries = load_queries('queries/queries.boolean.txt')

    # Execute Boolean queries for CW1
    results = ""
    for query_num, query in zip(queries_num, queries):
        results_tmp = execute_queries_and_save_results(query_num, query, search_type="boolean", indexer=indexer,
                                                       preprocessor=preprocessor, config=config)
        if results_tmp is not None:
            results = results + results_tmp

    with open("data/results/Boolean_" + '_results_queries.txt', mode="w", encoding="utf-8") as f:
        f.writelines(results[:-1])

    # Load ranked queries
    queries_num, queries = load_queries('queries/queries.ranked.txt')

    # Execute Ranked queries for CW1
    results = ""
    for query_num, query in zip(queries_num, queries):
        results_tmp = execute_queries_and_save_results(query_num, query, search_type="tfidf", indexer=indexer,
                                                       preprocessor=preprocessor, config=config)
        if results_tmp is not None:
            results = results + results_tmp
    with open("data/results/Ranked_" + '_results_queries.txt', mode="w", encoding="utf-8") as f:
        f.writelines(results[:-1])
    #
    # # Load boolean + ranked queries
    queries_num, queries = load_queries('queries/queries.boolean_and_ranked.txt')

    # Execute Boolean Ranked queries
    results = ""
    for query_num, query in zip(queries_num, queries):
        results_tmp, _ = execute_queries_and_save_results(query_num, query, search_type="boolean_and_tfidf", indexer=indexer,
                                                       preprocessor=preprocessor, config=config)
        if results_tmp is not None:
            results = results + results_tmp
    with open("data/results/Ranked_AND_Boolean" + '_results_queries.txt', mode="w", encoding="utf-8") as f:
        f.writelines(results[:-1])
    # #
    # # Load boolean + ranked queries
    queries_num, queries = load_queries('queries/queries.multiple_boolean_and_ranked.txt')

    # Execute Multiple Boolean Ranked queries
    results = ""
    for query_num, query in zip(queries_num, queries):
        results_tmp, _ = execute_queries_and_save_results(query_num, query, search_type="boolean_and_tfidf",
                                                       indexer=indexer,
                                                       preprocessor=preprocessor, config=config)
        if results_tmp is not None:
            results = results + results_tmp
    with open("data/results/Multiple_Ranked_AND_Boolean" + '_results_queries.txt', mode="w", encoding="utf-8") as f:
        f.writelines(results[:-1])

    # Load boolean + ranked queries
    queries_num, queries = load_queries('queries/queries.asterixForMissingWord.txt')

    # Execute Asterix Boolean Ranked queries
    results = ""
    for query_num, query in zip(queries_num, queries):
        results_tmp, _ = execute_queries_and_save_results(query_num, query, search_type="boolean_and_tfidf",
                                                       indexer=indexer,
                                                       preprocessor=preprocessor, config=config)
        if results_tmp is not None:
            results = results + results_tmp
    with open("data/results/asterixForMissingWord" + '_results_queries.txt', mode="w", encoding="utf-8") as f:
        f.writelines(results[:-1])

    dt_string_END = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    logging.warning(f'END date and time = {dt_string_END}')

    # Load boolean + ranked queries
    queries_num, queries = load_queries('queries/queries.system_evaluation.txt')

    # Execute Boolean Ranked queries
    results = ""
    results_data_frame = pd.DataFrame()
    for query_num, query in zip(queries_num, queries):
        results_tmp, results_data_frame_tmp = execute_queries_and_save_results(query_num, query,
                                                                               search_type="boolean_and_tfidf",
                                                                               indexer=indexer,
                                                                               preprocessor=preprocessor, config=config)
        if results_tmp is not None:
            results = results + results_tmp

        results_data_frame = results_data_frame.append(results_data_frame_tmp)

    if config["retrieval"]["perform_system_evaluation"]:
        dummy_correct_results = results_data_frame[["query_number", "doc_number", "score"]].reset_index(drop=True)
        dummy_correct_results["relevance"] = [round(x, 0) for x in dummy_correct_results["score"]]
        dummy_correct_results.drop(columns=["score"], inplace=True)
        dummy_correct_results.to_csv("correct_search_results.csv", index=False)
        print("Perform system evaluation")
        df_correct_search_results = pd.read_csv("correct_search_results.csv")
        df_correct_search_results["query_number"] = df_correct_search_results["query_number"].astype(str)

        # Calculate precision
        df_evaluation_results = calculate_precision(results_data_frame, df_correct_search_results, cutoff=5)

        # Calculate recall
        df_evaluation_results = pd.merge(df_evaluation_results,
                                         calculate_recall(results_data_frame, df_correct_search_results, cutoff=5),
                                         how="left",
                                         on=["query_number"])

        # Calculate AP
        df_evaluation_results = pd.merge(df_evaluation_results,
                                         calculate_average_precision(results_data_frame, df_correct_search_results),
                                         how="left",
                                         on=["query_number"])

        # Calculate nDCG@10
        df_evaluation_results = pd.merge(df_evaluation_results,
                                         calculate_discounted_cumulative_gain(results_data_frame,
                                                                              df_correct_search_results, rank=10),
                                         how="left",
                                         on=["query_number"])

        # Output results in the appropriate format
        df_evaluation_results = df_evaluation_results.round(3)
        df_evaluation_results.to_csv("data/results/" + 'results_system_evaluation.csv', index=False)

    with open("data/results/queries_system_evaluation.txt", mode="w", encoding="utf-8") as f:
        f.writelines(results[:-1])
