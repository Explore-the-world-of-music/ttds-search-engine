"""
This module contains the retrieval functions to return the search results
"""
from collections import defaultdict, Counter
import re
from helpers.misc import create_default_dict_list
import numpy as np
from operator import itemgetter
import pandas as pd
import datetime
import logging
import copy


def find_docs_with_term(term, index):
    """
    Returns all doc_ids which contain the "term".

    :param term: The searched term (str)
    :param index: Index in which to search (dict)
    :return: list of relevant doc ids (list)
    """
    if term in index.keys():
        rel_doc_ids = list(index[term].keys())
    else:
        rel_doc_ids = []

    return rel_doc_ids


def get_rel_doc_pos(term, index):
    """
    Returns all relevant doc_positions of a term

    :param term: The searched term (str)
    :param index: Index in which to search (dict)
    :return: positions of term for related doc ids (list of lists)
    """
    if term in index.keys():
        rel_doc_pos = index[term]
    else:
        rel_doc_pos = dict()

    return rel_doc_pos


def get_tfs_docs(term, index):
    """
    Returns term frequencies of a term in the documents

    :param term: The searched term (str)
    :param index: Index in which to search (dict)
    :return: term frequencies of term for related doc ids (dict)
    """
    if term in index.keys():

        tfs_docs = index[term].copy()
        tfs_docs = defaultdict(int, tfs_docs)
        for doc in tfs_docs.keys():
            tfs_docs[doc] = len(tfs_docs[doc])
    else:
        tfs_docs = dict()
        tfs_docs = defaultdict(int, tfs_docs)

    return tfs_docs


def get_tfs_docs_bool_search(search_results, bool_vals, indexer):
    """
    Returns term frequencies of a boolean query

    :param search_results: Results of document and tfs for each individual search term (dict)
    :param bool_vals: List of "&&", "&&--" or "||" or "||--" (list)
    :param indexer: Class instance for the created index (Indexer)
    :return: term frequencies which are relevant for this boolean search (dict)
    """
    terms = list(search_results.keys())
    # Extract tfs for the first term as basis
    tfs_docs = copy.deepcopy(search_results)

    for idx, bool_val in enumerate(bool_vals):

        # if bool_val in ["&&", "||"]: Nothing to change as all term frequencies matter

        if bool_val == "&&--":
            # Delete this query component as it adds no information about relevance of the individual query components
            del tfs_docs[terms[idx + 1]]

        elif bool_val == "||--":
            # Todo: Note improvement and correction here
            # Here we need to inverse the logic and return the documents that do not contain the term
            # rel_docs_new = sorted(set([doc_id for doc_id in indexer.all_doc_ids if doc_id not in tfs_docs[terms[idx + 1]]["rel_docs"]]))
            rel_docs_new = list(set(indexer.all_doc_ids) - set(tfs_docs[terms[idx + 1]]["rel_docs"]))

            # As this represents only a weak search results, the term frequency is set to 0.5 as documents
            # that have true positives should be favored
            tfs_docs_new = dict.fromkeys(rel_docs_new, 0.5)

            # Overwrite old values
            tfs_docs[terms[idx + 1]]["rel_docs"] = rel_docs_new
            tfs_docs[terms[idx + 1]]["tfs_docs"] = tfs_docs_new

    return tfs_docs


def bool_search(search_results, indexer, bool_vals):
    """
    Executes a boolean search between relevant documents for all searched terms.

    :param search_results: Results of document and tfs for each individual search term (dict)
    :param indexer: Class instance for the created index (Indexer)
    :param bool_vals: List of "&&", "&&--" or "||" or "||--" (list)
    :return: List of all doc_ids which are relevant for this boolean search (list)
    """
    terms = list(search_results.keys())
    # Extract relevant documents for the first term as basis for boolean analysis
    rel_docs = copy.deepcopy(search_results[terms[0]]["rel_docs"])

    for idx, bool_val in enumerate(bool_vals):
        if bool_val == "&&":
            # Todo: Note adjustment 1
            # rel_docs = [doc_id for doc_id in rel_docs if doc_id in search_results[terms[idx + 1]]["rel_docs"]]
            rel_docs = list(set(rel_docs).intersection(search_results[terms[idx + 1]]["rel_docs"]))

        elif bool_val == "&&--":
            # Todo: Note adjustment 2
            # rel_docs = [doc_id for doc_id in rel_docs if doc_id not in search_results[terms[idx + 1]]["rel_docs"]]
            rel_docs = list(set(rel_docs) - set(search_results[terms[idx + 1]]["rel_docs"]))

        elif bool_val == "||":
            rel_docs = list(set(rel_docs + search_results[terms[idx + 1]]["rel_docs"]))

        elif bool_val == "||--":
            # Todo: Note adjustment 3
            # rel_docs = list(set([doc_id for doc_id in range(1,indexer.total_num_docs+1) if doc_id not in
            #                    search_results[terms[idx + 1]]["rel_docs"]] + rel_docs))
            rel_docs = list((set(indexer.all_doc_ids) -
                             set(search_results[terms[idx + 1]]["rel_docs"])).union(set(rel_docs)))

        else:
            raise Exception(
                "bool_val of search doesn't match either '&&', '&&--', '||' or '||--'. It is: {}.".format(
                    bool_val))

    return rel_docs


def simple_proximity_search(search_results, indexer, n=1, pos_asterisk=None, phrase=False):
    """
    Calculates if terms in query are in the same document with less or equal to n distance and return relevant doc_ids.
    Option to perform phrase search.

    :param search_results: Results of document and tfs for each individual search term (dict)
    :param indexer: Class instance for the created index (Indexer)
    :param n: allowed distance in one document (int)
    :param pos_asterisk: positions where the asterisks are located (list)
    :param phrase: whether or not the search is a phrase search and ordering matters (bool)
    :return: List of all doc_ids which are relevant for proximity search (list)
    """
    # Performs a boolean search to get documents which contain both terms
    terms = list(search_results.keys())
    rel_documents_all_terms = bool_search(search_results, indexer=indexer, bool_vals=["&&"] * (len(terms) - 1))

    # if any(|pos_1 - pos_2|<= n) --> doc_id is relevant --> append it to returned final_rel_doc_ids list
    # Find potential candidates (differentiation important for multiple words in phrase/proximity search)
    final_rel_doc_ids = list()
    for doc_id in rel_documents_all_terms:
        dict_candi = defaultdict(list)
        for idx, term in enumerate(terms[:-1]):
            for pos_1 in search_results[terms[idx]]["rel_doc_pos"][doc_id]:
                for pos_2 in search_results[terms[idx + 1]]["rel_doc_pos"][doc_id]:
                    if phrase:
                        diff = 1
                        if pos_asterisk is not None:
                            if (idx + 1) in pos_asterisk:
                                diff = 1 + len([num for num in pos_asterisk if num == idx + 1])
                        if int(pos_2) - int(pos_1) == diff:
                            dict_candi[idx].append((pos_1, pos_2))
                    else:
                        if abs(int(pos_1) - int(pos_2)) <= n:
                            dict_candi[idx].append((pos_1, pos_2))

        # Check if found candidates are actual true positives
        if len(terms) <= 2:
            # If only two terms were compared, take all results
            for _ in dict_candi[0]:
                final_rel_doc_ids.append(doc_id)
        else:
            # For multiple terms delete all positions which are true across terms
            # Finally take the minimum true positions per term as the number for which the whole
            # phrase or proximity was found in the document
            dict_candi_fin = copy.deepcopy(dict_candi)
            for idx in np.arange(len(terms[:-1]) - 1):
                counter = 0
                for idx2, candi_1 in enumerate(dict_candi[idx]):
                    candi_2 = [pos[0] for pos in dict_candi[idx + 1]]
                    if candi_1[1] not in candi_2:
                        del dict_candi_fin[idx][idx2 - counter]
                        counter = counter + 1
            if len(list(dict_candi_fin.keys())) > 0:
                max_cand = min([len(dict_candi_fin[key]) for key in dict_candi.keys()])
                for i in np.arange(max_cand):
                    final_rel_doc_ids.append(doc_id)

    # Convert results to appropriate output format
    tfs_docs = dict(Counter(final_rel_doc_ids))
    tfs_docs = defaultdict(int, tfs_docs)
    final_rel_doc_ids = list(set(final_rel_doc_ids))

    return final_rel_doc_ids, tfs_docs


def simple_tfidf_search(terms, indexer):
    """
    Calculates the TF-IDF score for multiple terms and returns an ordered dict.

    :param terms: List of terms contained in search (list)
    :param indexer: Class instance for the created index (Indexer)
    :return: Descending sorted pseudo-dictionary with doc_id as key and TF-IDF as value (list)
    """

    doc_relevance = defaultdict(lambda: 0)
    total_num_docs = len(indexer.all_doc_ids)

    for t in terms:
        # Todo: Note optimizations here
        # rel_docs = find_docs_with_term(t, indexer.index)
        # df = len(rel_docs
        # rel_doc_pos = get_rel_doc_pos(t, indexer.index)
        # tfs_docs = [len(rel_doc_pos[key]) for key in rel_doc_pos]
        # weights_docs = [(1 + np.log10(tf)) * np.log10(total_num_docs / df) for tf in tfs_docs]

        logging.info(f'Calculations for {t}')
        TIMESTAMP = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S:%f")
        logging.info(f'TIMESTAMP 1 = {TIMESTAMP}')

        tfs_docs = get_tfs_docs(t, indexer.index)
        rel_docs = list(tfs_docs.keys())
        df = len(rel_docs)

        if df > 0:

            TIMESTAMP = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S:%f")
            logging.info(f'TIMESTAMP 2 = {TIMESTAMP}')

            if total_num_docs == df:
                scale = 1
            else:
                scale = np.log10(total_num_docs / df)

            # Todo: Note optimization here
            # weights_docs = [(1 + np.log10(tfs_docs[key])) * scale for key in rel_docs]
            weights_docs = [(1 + np.log10(value)) * scale for key, value in tfs_docs.items()]

        else:
            weights_docs = []

        TIMESTAMP = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S:%f")
        logging.info(f'TIMESTAMP 3 = {TIMESTAMP}')

        # Todo: Note optimization
        for doc_id, weight in zip(rel_docs, weights_docs):
            doc_relevance[doc_id] += weight

        # for doc_id, weight in zip(rel_docs, weights_docs):
        #     if doc_id not in doc_relevance:
        #         doc_relevance[doc_id] = weight
        #     else:
        #         doc_relevance[doc_id] += weight

    sorted_relevance = sorted(doc_relevance.items(), key=lambda x: float(x[0]), reverse=False)
    sorted_relevance = sorted(sorted_relevance, key=lambda x: x[1], reverse=True)

    return sorted_relevance


def calculate_tfidf(rel_docs, tfs_docs, indexer, logical_search):
    """
    Calculates the TF-IDF score for given search results for one search term

    :param rel_docs: List of relevant documents (list)
    :param tfs_docs: Term frequency in the relevant documents (list)
    :param indexer: Class instance for the created index (Indexer)
    :param logical_search: Boolean value if the search is logical
    :return: Descending sorted dictionary with doc_id as key and TF-IDF as value (dict)
    """
    doc_relevance = {}
    total_num_docs = len(indexer.all_doc_ids)

    # Split cases for boolean search and searches with only one query component
    if logical_search:

        for query_component in tfs_docs.keys():
            logging.info(f'Calculations for {query_component}')
            TIMESTAMP = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            logging.info(f'TIMESTAMP = {TIMESTAMP}')
            # Extract the document frequency for the query component
            rel_docs_all = tfs_docs[query_component]["rel_docs"]
            df = len(rel_docs_all)

            if df > 0:
                # Todo: Note optimization here
                # Extract the query component frequencies but only for the RELEVANT documents
                # tfs_docs_all = [tfs_docs[query_component]["tfs_docs"][key] for key in rel_docs_all if key in rel_docs]
                if total_num_docs == df:
                    scale = 1
                else:
                    scale = np.log10(total_num_docs / df)
                docs_loop = list(set(tfs_docs[query_component]["tfs_docs"].keys()).intersection(rel_docs))
                tfs_docs_all = [tfs_docs[query_component]["tfs_docs"][key] for key in docs_loop]

                # Sum over all relevant documents
                weights_docs = [(1 + np.log10(tf)) * scale for tf in tfs_docs_all]
            else:
                docs_loop = []
                weights_docs = []

            # Todo: Note optimization here
            # rel_docs_fin = [doc_id for doc_id in rel_docs_all if doc_id in rel_docs]
            for doc_id, weight in zip(docs_loop, weights_docs):
                if doc_id not in doc_relevance:
                    doc_relevance[doc_id] = weight
                else:
                    doc_relevance[doc_id] += weight

        # Sort values
        sorted_relevance = sorted(doc_relevance.items(), key=lambda x: float(x[0]), reverse=False)
        sorted_relevance = sorted(sorted_relevance, key=lambda x: x[1], reverse=True)
    else:
        # Only one search component
        df = len(rel_docs)  # document frequency

        # Calculate the weights per document
        if df > 0:
            scale = np.log10(total_num_docs / df)
            # Please note that the calculation was adjusted to differentiate documents in ranking even if
            # all documents of the collection are part of the relevant documents
            if total_num_docs == df:
                weights_docs = [(1 + np.log10(tfs_docs[key])) * 1 for key in rel_docs]
            else:
                weights_docs = [(1 + np.log10(tfs_docs[key])) * scale for key in rel_docs]
        else:
            weights_docs = []

        for doc_id, weight in zip(rel_docs, weights_docs):
            doc_relevance[doc_id] = weight

        # Sort values
        sorted_relevance = sorted(doc_relevance.items(), key=lambda x: float(x[0]), reverse=False)
        sorted_relevance = sorted(sorted_relevance, key=lambda x: x[1], reverse=True)

    return sorted_relevance


def execute_search(query, indexer, preprocessor):
    """
    Checks, which type of search has to be done.
    Executes the search and returns the resulting, relevant doc_ids and tfs for those

    :param query: The query which should be searched for (str)
    :param indexer: Class instance for the created index (Indexer)
    :param preprocessor: Preprocessor class instance (Preprocessor)
    :return: List from the matching function containing all relevant doc_ids, List of all tfs for relevant doc_ids
    """
    # compile search patterns to test for
    bool_pattern = re.compile(r"(&&--)|(\|\|--)|(&&)|(\|\|)")
    prox_pattern = re.compile(r"#\d+")
    phra_pattern = re.compile(r'^".*"$')

    # check if boolean search
    if bool_pattern.search(query) is not None:

        type_of_bool_search = bool_pattern.findall(query)
        for idx, item in enumerate(type_of_bool_search):
            type_of_bool_search[idx] = [x for x in item if x != ""][0].strip()

        terms = list()
        pos_bool_old = 0
        for idx, bool in enumerate(type_of_bool_search):
            pos_bool_new = query[pos_bool_old:].find(type_of_bool_search[idx]) + pos_bool_old
            terms.append(query[pos_bool_old:pos_bool_new].strip())
            pos_bool_old = pos_bool_new + len(bool)
        terms.append(query[pos_bool_old:].strip())

        search_results = defaultdict(create_default_dict_list)
        for idx, term in enumerate(terms):
            key = term + "_" + str(idx)
            search_results[key]["rel_docs"], search_results[key]["tfs_docs"] = execute_search(term, indexer,
                                                                                              preprocessor)

        rel_docs = bool_search(search_results, indexer=indexer, bool_vals=type_of_bool_search)
        tfs_docs = get_tfs_docs_bool_search(search_results, bool_vals=type_of_bool_search, indexer=indexer)

        return rel_docs, tfs_docs

    # check if proximity search
    elif prox_pattern.search(query) is not None:
        n = int(prox_pattern.search(query).group(0)[1:])
        terms = [re.sub('[^a-zA-Z]+', '', term) for term in query.split(",")]

        search_results = defaultdict(create_default_dict_list)
        for idx, term in enumerate(terms):
            key = term + "_" + str(idx)
            # Todo: Note optimization here
            # search_results[term]["rel_docs"], _ = execute_search(term, indexer, preprocessor)
            search_results[key]["rel_doc_pos"] = get_rel_doc_pos(preprocessor.preprocess(term)[0], indexer.index)
            search_results[key]["rel_docs"] = list(search_results[key]["rel_doc_pos"].keys())

        rel_docs, tfs_docs = simple_proximity_search(search_results, indexer=indexer, n=n)

        return rel_docs, tfs_docs

    # check if phrase search --> same as proximity search with n = 1
    elif phra_pattern.search(query) is not None:
        terms = re.sub('"', "", query).split()

        # Check if asterisk is in query
        if "*" in terms:
            pos_asterisk = list()
            for idx, term in enumerate(terms):
                if term == "*":
                    if len(pos_asterisk) > 0:
                        if pos_asterisk[-1] == (idx - 1):
                            pos_asterisk.append(idx - 1)
                        else:
                            pos_asterisk.append(idx - len(pos_asterisk))
                    else:
                        pos_asterisk.append(idx - len(pos_asterisk))

            terms = [term for term in terms if term != "*"]
        else:
            pos_asterisk = None

        search_results = defaultdict(create_default_dict_list)
        for idx, term in enumerate(terms):
            # Todo: Note optimization here
            # search_results[term]["rel_docs"], _ = execute_search(term, indexer, preprocessor)
            key = term + "_" + str(idx)
            search_results[key]["rel_doc_pos"] = get_rel_doc_pos(preprocessor.preprocess(term)[0], indexer.index)
            search_results[key]["rel_docs"] = list(search_results[key]["rel_doc_pos"].keys())

        if len(search_results.keys()) < 2:
            # Todo: Note optimization here
            key = list(search_results.keys())[0]
            final_rel_doc_ids = search_results[key]["rel_docs"]
            # Convert results to appropriate output format
            tfs_docs = defaultdict(int, dict(Counter(final_rel_doc_ids)))
            for key2 in tfs_docs.keys():
                tfs_docs[key2] = len(search_results[key]["rel_doc_pos"][key2])

            return final_rel_doc_ids, tfs_docs
        else:
            rel_docs, tfs_docs = simple_proximity_search(search_results, indexer=indexer, n=1, phrase=True,
                                                         pos_asterisk=pos_asterisk)
            return rel_docs, tfs_docs

    # if nothing else matches --> make a simple search
    else:
        # Todo: Note optimization here
        tfs_docs = get_tfs_docs(preprocessor.preprocess(query)[0], indexer.index)
        # results = find_docs_with_term(preprocessor.preprocess(query)[0], indexer.index)
        results = list(tfs_docs.keys())

        return results, tfs_docs


def execute_queries_and_save_results(query_num, query, search_type, indexer, preprocessor, config):
    """
    Function to execute search and return results
    :param query_num: Number of query (int)
    :param query: Query that should be searched (str)
    :param search_type: Parameter which search type it is (str)
    :param indexer: Class instance for the created index (Indexer)
    :param preprocessor: Preprocessor class instance (Preprocessor)
    :param config: Defined configuration settings (dict)
    :return: results (str)
    """
    # Initiate result string
    results = ""

    # Execute search for boolean queries
    if search_type == "boolean":
        rel_docs, _ = execute_search(query, indexer, preprocessor)
        if len(rel_docs) > 0:
            rel_docs.sort(key=float)
            for rel_doc in rel_docs:
                results += f"{query_num},{rel_doc}\n"

        return results

    # Execute search for ranked queries
    if search_type == "tfidf":
        terms = query.split()
        terms = [preprocessor.preprocess(term)[0] for term in terms if len(preprocessor.preprocess(term)) > 0]
        rel_docs = simple_tfidf_search(terms, indexer)

        if len(rel_docs) > 0:
            if len(rel_docs) > config["retrieval"]["number_ranked_documents"]:
                rel_docs = rel_docs[:config["retrieval"]["number_ranked_documents"]]
            for doc_id, value in rel_docs:
                results += f"{query_num},{doc_id},{round(value, 4)}\n"

            return results

    if search_type == "boolean_and_tfidf":
        # Execute search for boolean queries considering ranking
        # boolean_search_pattern = re.compile('(\sAND NOT\s)|(\sOR NOT\s)|(\sAND\s)|(\sOR\s)|(#\d+)|^".*"$')
        search_pattern = re.compile('(&&--)|(\|\|--)|(&&)|(\|\|)|(#\d+)|^".*"$')
        bool_pattern = re.compile(r"(&&--)|(\|\|--)|(&&)|(\|\|)")

        if bool_pattern.search(query) is not None:
            logical_search = True
        else:
            logical_search = False

        # check if boolean search component is in query
        if search_pattern.search(query) is not None:
            rel_docs, tfs_docs = execute_search(query, indexer, preprocessor)
            rel_docs_with_tfidf = calculate_tfidf(rel_docs, tfs_docs, indexer, logical_search)
        else:
            terms = query.split()
            terms = [preprocessor.preprocess(term)[0] for term in terms if len(preprocessor.preprocess(term)) > 0]
            rel_docs_with_tfidf = simple_tfidf_search(terms, indexer)

        if len(rel_docs_with_tfidf) > 0:
            # Only keep top results
            if len(rel_docs_with_tfidf) > config["retrieval"]["number_ranked_documents"]:
                rel_docs_with_tfidf = rel_docs_with_tfidf[:config["retrieval"]["number_ranked_documents"]]

            # Rescale the results. For queries with "OR NOT" it can happen that the difference in scores between the
            # documents are very low (0.0001). To interpret results easier we re-scale here based on the highest score
            max_value = max(rel_docs_with_tfidf, key=itemgetter(1))[1]
            rel_docs_with_tfidf_scaled = list()
            for idx, _ in enumerate(rel_docs_with_tfidf):
                rel_docs_with_tfidf_scaled.append((rel_docs_with_tfidf[idx][0], rel_docs_with_tfidf[idx][1] / max_value * 10))

            # Write output
            for doc_id, value in rel_docs_with_tfidf_scaled:
                results += f"{query_num},{doc_id},{round(value, 4)}\n"

            if config["retrieval"]["perform_system_evaluation"]:
                doc_number = [x[0] for x in rel_docs_with_tfidf_scaled]
                rank_of_doc = np.arange(1, len(doc_number) +1)
                score = [x[1] for x in rel_docs_with_tfidf_scaled]
                query_number =  [query_num] * len(doc_number)

                results_frame = pd.DataFrame({"query_number":query_number,"doc_number":doc_number,"rank_of_doc":rank_of_doc,
                                              "score":score})
            else:
                results_frame = pd.DataFrame()

            return results, results_frame

        else:

            results_frame = pd.DataFrame()
            return results, results_frame
