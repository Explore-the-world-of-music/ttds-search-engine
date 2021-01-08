"""
This module creates the retrieval class to return the search results
"""
import re
import collections
import numpy as np


def find_docs_with_term(term, index):
    """
    Returns all doc_ids which contain the "term".

    :param term: The searched term (str)
    :param index: Index in which to search (dict)
    :return: list of relevant doc ids (list)
    """
    try:
        rel_doc_ids = list(index[term].keys())
    except:
        rel_doc_ids = []

    return rel_doc_ids


def get_rel_doc_pos(term, index):
    """
    Returns all relevant doc_positions of a term

    :param term: The searched term (str)
    :param index: Index in which to search (dict)
    :return: positions of term for related doc ids (list of lists)
    """
    docs_info = index[term]
    return [docs_info[doc] for doc in docs_info.keys()]


def get_tfs_docs(term, index):
    """
    Returns term frequencies of a term in the documents

    :param term: The searched term (str)
    :param index: Index in which to search (dict)
    :return: term frequencies of term for related doc ids (dict)
    """
    try:
        tfs_docs = index[term].copy()
        tfs_docs = collections.defaultdict(int, tfs_docs)
        for doc in tfs_docs.keys():
            tfs_docs[doc] = len(tfs_docs[doc])
    except:
        tfs_docs = dict()

    return tfs_docs

def get_tfs_docs_bool_search(rel_docs, tfs_docs_t1, tfs_docs_t2):
    """
    Returns term frequencies of a boolean query

    :param rel_docs: List of all doc_ids which are relevant for this boolean search (list)
    :param tfs_docs_t1: Term frequencies for term 1 (dict)
    :param tfs_docs_t2: Term frequencies for term 2 (dict)
    :return: term frequencies which are relevant for this boolean search (dict)
    """
    tfs_docs = [tfs_docs_t1[doc] + tfs_docs_t2[doc] for doc in rel_docs]

    return tfs_docs


def simple_bool_search(rel_docs_t1, rel_docs_t2, indexer, bool_val="AND"):
    """
    Executes a simple boolean search between two lists of relevant docs.

    :param rel_docs_t1: Relevant doc_ids of term 1 (list)
    :param rel_docs_t2: Relevant doc_ids of term 2 (list)
    :param indexer: Class instance for the created index (Indexer)
    :param bool_val: Either "AND", "AND NOT" or "OR" (str)
    :return: List of all doc_ids which are relevant for this boolean search (list)
    """
    if bool_val == "AND":
        return [doc_id for doc_id in rel_docs_t1 if doc_id in rel_docs_t2]

    elif bool_val == "AND NOT":
        return [doc_id for doc_id in rel_docs_t1 if doc_id not in rel_docs_t2]

    elif bool_val == "OR":
        return list(set(rel_docs_t1 + rel_docs_t2))

    elif bool_val == "OR NOT":
        return list(
            set([doc_id for doc_id in indexer.all_doc_ids if doc_id not in rel_docs_t2] + rel_docs_t1))

    else:
        raise Exception(
            "bool_val of simple_bool_search doesn't match either 'AND', 'AND NOT', 'OR' or 'OR NOT'. It is: {}.".format(
                bool_val))


def simple_proximity_search(rel_docs_t1, rel_doc_pos_t1, rel_docs_t2, rel_doc_pos_t2, indexer, n=1, phrase=False):
    """
    Calculates if term1 and term2 are in the same document with less or equal to n distance and return relevant doc_ids.
    Option to perform phrase search.

    :param rel_docs_t1: Relevant doc_ids of term 1 (list)
    :param rel_doc_pos_t1: Relevant positions, referring to the doc_ids of term 1 (list of lists)
    :param rel_docs_t2: Relevant doc_ids of term 2 (list)
    :param rel_doc_pos_t2: Relevant positions, referring to the doc_ids of term 2 (list of lists)
    :param indexer: Class instance for the created index (Indexer)
    :param n: allowed distance in one document (int)
    :param phrase: whether or not the search is a phrase search and ordering matters (bool)
    :return: List of all doc_ids which are relevant for proximity search (list)
    """
    # Performs a boolean search to get documents which contain both terms
    rel_documents_both_terms = simple_bool_search(rel_docs_t1, rel_docs_t2, indexer=indexer, bool_val="AND")

    # Create boolean masks to match extract the relevant doc_pos parts
    bool_mask_t1 = [True if doc in rel_documents_both_terms else False for doc in rel_docs_t1]
    bool_mask_t2 = [True if doc in rel_documents_both_terms else False for doc in rel_docs_t2]

    rel_doc_pos_t1 = [pos for pos, boolean in zip(rel_doc_pos_t1, bool_mask_t1) if boolean]
    rel_doc_pos_t2 = [pos for pos, boolean in zip(rel_doc_pos_t2, bool_mask_t2) if boolean]

    # For each document id
    # if any(|pos_1 - pos_2|<= n) --> doc_id is relevant --> append it to returned final_rel_doc_ids list
    final_rel_doc_ids = []
    for idx, doc_id in enumerate(rel_documents_both_terms):

        for pos_1 in rel_doc_pos_t1[idx]:

            for pos_2 in rel_doc_pos_t2[idx]:
                if phrase:
                    if int(pos_2) - int(pos_1) == n:
                        final_rel_doc_ids.append(doc_id)
                        break

                else:
                    if abs(int(pos_1) - int(pos_2)) <= n:
                        final_rel_doc_ids.append(doc_id)
                        break

    tfs_docs = dict(collections.Counter(final_rel_doc_ids))
    final_rel_doc_ids = sorted(list(set(final_rel_doc_ids)))

    return final_rel_doc_ids, tfs_docs


def simple_tfidf_search(terms, indexer):
    """
    Calculates the TF-IDF score for multiple terms and returns an ordered dict.

    :param terms: List of terms contained in search (list)
    :param indexer: Class instance for the created index (Indexer)
    :return: Descending sorted pseudo-dictionary with doc_id as key and TF-IDF as value (list)
    """
    doc_relevance = {}
    total_num_docs = len(indexer.all_doc_ids)

    for t in terms:
        rel_docs = find_docs_with_term(t, indexer.index)
        df = len(rel_docs)
        tfs_docs = [len(doc) for doc in get_rel_doc_pos(t, indexer.index)]
        weights_docs = [(1 + np.log10(tf)) * np.log10(total_num_docs / df) for tf in tfs_docs]

        for doc_id, weight in zip(rel_docs, weights_docs):

            if doc_id not in doc_relevance:
                doc_relevance[doc_id] = weight
            else:
                doc_relevance[doc_id] += weight

    sorted_relevance = sorted(doc_relevance.items(), key=lambda x: x[1], reverse=True)
    return sorted_relevance


def calculate_tfidf(rel_docs, tfs_docs, indexer):
    """
    Calculates the TF-IDF score for given search results for one search term

    :param rel_docs: List of relevant documents (list)
    :param tfs_docs: Term frequency in the relevant documents (list)
    :param indexer: Class instance for the created index (Indexer)
    :return: Descending sorted dictionary with doc_id as key and TF-IDF as value (dict)
    """
    doc_relevance = {}
    total_num_docs = len(indexer.all_doc_ids)
    df = len(rel_docs) # document frequency

    # Calculate the weights per document
    weights_docs = [(1 + np.log10(tf)) * np.log10(total_num_docs / df) for tf in tfs_docs]

    for doc_id, weight in zip(rel_docs, weights_docs):
        if doc_id not in doc_relevance:
            doc_relevance[doc_id] = weight
        else:
            doc_relevance[doc_id] += weight

    sorted_relevance = sorted(doc_relevance.items(), key=lambda x: x[1], reverse=True)
    return sorted_relevance


def execute_search(query, indexer, preprocessor):
    """
    Checks, which type of search has to be done.
    Executes the search and returns the resulting, relevant doc_ids

    :param query: The query which should be searched for (str)
    :param indexer: Class instance for the created index (Indexer)
    :param preprocessor: Preprocessor class instance (Preprocessor)
    :return: List from the matching function containing all relevant doc_ids
    """
    # compile test patterns
    bool_pattern = re.compile("(\sAND NOT\s)|(\sOR NOT\s)|(\sAND\s)|(\sOR\s)")
    prox_pattern = re.compile("#\d+")
    phra_pattern = re.compile('^".*"$')

    # check if boolean search
    if bool_pattern.search(query) != None:
        type_of_bool_search = bool_pattern.search(query).group(0)
        t1, t2 = query.split(type_of_bool_search)

        rel_docs_t1, tfs_docs_t1 = execute_search(t1, indexer, preprocessor)
        rel_docs_t2, tfs_docs_t2 = execute_search(t2, indexer, preprocessor)
        rel_docs = simple_bool_search(rel_docs_t1, rel_docs_t2, indexer=indexer, bool_val=type_of_bool_search.strip())
        tfs_docs = get_tfs_docs_bool_search(rel_docs, tfs_docs_t1, tfs_docs_t2)

        return rel_docs, tfs_docs

    # check if proximity search
    elif prox_pattern.search(query) != None:
        n = int(prox_pattern.search(query).group(0)[1:])
        t1, t2 = [re.sub('[^a-zA-Z]+', '', term) for term in query.split(",")]

        rel_docs_t1, _ = execute_search(t1, indexer, preprocessor)
        rel_docs_t2, _ = execute_search(t2, indexer, preprocessor)
        rel_doc_pos_t1 = get_rel_doc_pos(preprocessor.preprocess(t1)[0], indexer.index)
        rel_doc_pos_t2 = get_rel_doc_pos(preprocessor.preprocess(t2)[0], indexer.index)

        rel_docs, tfs_docs = simple_proximity_search(rel_docs_t1, rel_doc_pos_t1, rel_docs_t2, rel_doc_pos_t2,
                                                     indexer=indexer, n=n)
        return rel_docs, tfs_docs

    # check if phrase search --> same as proximity search with n = 1
    elif phra_pattern.search(query) != None:
        t1, t2 = re.sub('"', "", query).split()

        rel_docs_t1, _ = execute_search(t1, indexer, preprocessor)
        rel_docs_t2, _ = execute_search(t2, indexer, preprocessor)
        rel_doc_pos_t1 = get_rel_doc_pos(preprocessor.preprocess(t1)[0], indexer.index)
        rel_doc_pos_t2 = get_rel_doc_pos(preprocessor.preprocess(t2)[0], indexer.index)

        rel_docs, tfs_docs = simple_proximity_search(rel_docs_t1, rel_doc_pos_t1, rel_docs_t2,
                                                     rel_doc_pos_t2, indexer=indexer, n=1, phrase=True)
        return rel_docs, tfs_docs

    # if nothing else matches --> make a simple search
    else:
        results = find_docs_with_term(preprocessor.preprocess(query)[0], indexer.index)
        tfs_docs = get_tfs_docs(preprocessor.preprocess(query)[0], indexer.index)

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

    # Todo: Take out boolean search type
    # Execute search for boolean queries
    if search_type == "boolean":
        rel_docs, _ = execute_search(query, indexer, preprocessor)
        if len(rel_docs) > 0:
            rel_docs.sort(key=float)
            for rel_doc in rel_docs:
                results += f"{query_num},{rel_doc}\n"

        return results

    # Todo: Take out boolean tfidf type
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
        # Execute search for boolean queries
        rel_docs, tfs_docs = execute_search(query, indexer, preprocessor)

        rel_docs_with_tfidf = calculate_tfidf(rel_docs, tfs_docs, indexer)

        if len(rel_docs_with_tfidf) > 0:
            if len(rel_docs_with_tfidf) > config["retrieval"]["number_ranked_documents"]:
                rel_docs = rel_docs[:config["retrieval"]["number_ranked_documents"]]
            for doc_id, value in rel_docs_with_tfidf:
                results += f"{query_num},{doc_id},{round(value, 4)}\n"
            return results