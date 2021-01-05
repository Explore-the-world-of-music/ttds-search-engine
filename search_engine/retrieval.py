"""
This module creates the retrieval class to get the search results
"""
import re
import numpy as np

def find_docs_with_term(term, index):
    '''
    Returns all doc_ids which contain the "term".

    Parameters:
        term (str): The searched term
        (optional) inv_index (dic): Inverted index in which to search

    Returns:
        rel_doc_ids (list): List of all doc_ids of documents, which contain "term"
    '''

    docs_info = index[term]
    rel_doc_ids = [doc for doc in docs_info.keys()]

    return rel_doc_ids


def get_rel_doc_pos(term, index):
    '''
    Returns all relevant doc_positions of a term

    Parameters:
        term (str): The searched term
        (optional) inv_index (dic): Inverted index in which to search

    Returns:
        rel_doc_pos (list of lists): each lists contains the position the term was found in the related doc_id
    '''

    docs_info = index[term]
    return [docs_info[doc] for doc in docs_info.keys()]


def simple_bool_search(rel_docs_t1, rel_docs_t2, indexer, bool_val="AND"):
    '''
    Executes a simple boolean search between two lists of relevant docs.

    Parameters:
        rel_docs_t1 (list of ints): Relevant doc_ids of term 1
        rel_docs_t2 (list of ints): Relevant doc_ids of term 2
        bool_val (str): Either "AND", "AND NOT" or "OR"

    Returns:
        (list): List of all doc_ids which are relevant for this boolean search
    '''

    if bool_val == "AND":
        return [doc_id for doc_id in rel_docs_t1 if doc_id in rel_docs_t2]

    elif bool_val == "AND NOT":
        return [doc_id for doc_id in rel_docs_t1 if doc_id not in rel_docs_t2]

    elif bool_val == "OR":
        return list(set(rel_docs_t1 + rel_docs_t2))

    elif bool_val == "OR NOT":
        print("here")
        return list(
            set([doc_id for doc_id in indexer.all_doc_ids if doc_id not in rel_docs_t2] + rel_docs_t1))

    else:
        raise Exception(
            "bool_val of simple_bool_search doesn't match either 'AND', 'AND NOT', 'OR' or 'OR NOT'. It is: {}.".format(
                bool_val))


def simple_proximity_search(rel_docs_t1, rel_doc_pos_t1, rel_docs_t2, rel_doc_pos_t2, indexer, n=1, phrase=False):
    '''
    Calculates if term1 and term2 are in the same document with less or equal to n distance and return relevant doc_ids. Option to perform phrase search.

    Parameters:
        rel_docs_t1 (list of ints): Relevant doc_ids of term 1
        rel_doc_pos_t1 (list of lists of ints): Relevant positions, referring to the doc_ids of term 1
        rel_docs_t2 (list of ints): Relevant doc_ids of term 2
        rel_doc_pos_t2 (list of lists of ints): Relevant positions, referring to the doc_ids of term 2
        n (int): allowed distance in one document
        phrase (bool): wether or not the search is a phrase search (ordering matters)

    Returns:
        final_rel_doc_ids (list): List of all doc_ids which are relevant for proximity search
    '''

    # Performs a boolean search to get documents which contain both terms
    rel_documents_both_terms = simple_bool_search(rel_docs_t1, rel_docs_t2, indexer=indexer, bool_val="AND")

    # create boolean masks to match extracte the relevant doc_pos parts
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

    return list(set(final_rel_doc_ids))


def simple_tfidf_search(terms, indexer):
    '''
    Calculates the TF-IDF score for multiple terms and returns an ordered dict.

    Parameters:
        terms (list): List of terms contained in search

    Returns:
        sorted_relevance (dic): Descenting sorted dictionary with doc_id as key and TF-IDF as value
    '''

    doc_relevance = {}
    total_num_docs = len(indexer.all_doc_ids)

    for t in terms:
        df = len(indexer.index[t].keys())
        rel_docs = find_docs_with_term(t, indexer.index)
        tfs_docs = [len(doc) for doc in get_rel_doc_pos(t,indexer.index)]
        weights_docs = [(1 + np.log10(tf)) * np.log10(total_num_docs / df) for tf in tfs_docs]

        for doc_id, weight in zip(rel_docs, weights_docs):

            if doc_id not in doc_relevance:
                doc_relevance[doc_id] = weight
            else:
                doc_relevance[doc_id] += weight

    sorted_relevance = sorted(doc_relevance.items(), key=lambda x: x[1], reverse=True)
    return sorted_relevance

def execute_search(query, indexer, preprocessor):
    '''
    Checks, which type of search has to be done. Executes the search and returns the resulting, relevant doc_ids

    Parameters:
        query (string): The query which should be searched for

    Returns:
        (list): List from the matching function containing all relevant doc_ids
    '''
    # compile test patterns
    bool_pattern = re.compile("(\sAND NOT\s)|(\sOR NOT\s)|(\sAND\s)|(\sOR\s)")
    prox_pattern = re.compile("#\d+")
    phra_pattern = re.compile('^".*"$')

    # check if boolean search
    if bool_pattern.search(query) != None:
        type_of_bool_search = bool_pattern.search(query).group(0)
        t1, t2 = query.split(type_of_bool_search)

        rel_docs_t1 = execute_search(t1, indexer, preprocessor)
        rel_docs_t2 = execute_search(t2, indexer, preprocessor)
        return simple_bool_search(rel_docs_t1, rel_docs_t2, indexer = indexer, bool_val=type_of_bool_search.strip())

    # check if proximity search
    elif prox_pattern.search(query) != None:
        n = int(prox_pattern.search(query).group(0)[1:])
        t1, t2 = [re.sub('[^a-zA-Z]+', '', term) for term in query.split(",")]

        rel_docs_t1 = execute_search(t1, indexer, preprocessor)
        rel_docs_t2 = execute_search(t2, indexer, preprocessor)
        rel_doc_pos_t1 = get_rel_doc_pos(preprocessor.preprocess(t1)[0], indexer.index)
        rel_doc_pos_t2 = get_rel_doc_pos(preprocessor.preprocess(t2)[0], indexer.index)
        return simple_proximity_search(rel_docs_t1, rel_doc_pos_t1, rel_docs_t2, rel_doc_pos_t2, indexer = indexer, n = n)

    # check if phrase search --> same as priximity search with n = 1
    elif phra_pattern.search(query) != None:
        t1, t2 = re.sub('"', "", query).split()

        rel_docs_t1 = execute_search(t1, indexer, preprocessor)
        rel_docs_t2 = execute_search(t2, indexer, preprocessor)
        rel_doc_pos_t1 = get_rel_doc_pos(preprocessor.preprocess(t1)[0], indexer.index)
        rel_doc_pos_t2 = get_rel_doc_pos(preprocessor.preprocess(t2)[0], indexer.index)
        return simple_proximity_search(rel_docs_t1, rel_doc_pos_t1, rel_docs_t2, rel_doc_pos_t2, indexer = indexer, n = 1, phrase = True)

    # if nothing else matches --> make a simple search
    else:
        results = find_docs_with_term(preprocessor.preprocess(query)[0], indexer.index)
        return results

def execute_queries_and_save_results(filepath, query_num, query, bool, indexer, preprocessor):

    text = ""
    if bool:
        rel_docs = execute_search(query, indexer, preprocessor)
        if len(rel_docs) > 0:
            rel_docs.sort(key = float)
            for rel_doc in rel_docs:
                text += f"{query_num},{rel_doc}\n"

        with open(filepath, mode = "w", encoding = "utf-8") as f:
            f.writelines(text[:-1])

    else:
        terms = query.split()
        terms = [preprocessor.preprocess(term)[0] for term in terms if len(preprocessor.preprocess(term))>0]
        rel_docs = simple_tfidf_search(terms, indexer)

        if len(rel_docs) > 0:
            if len(rel_docs) > 150:
                rel_docs = rel_docs[:150]
            for doc_id, value in rel_docs:
                text += f"{query_num},{doc_id},{round(value,4)}\n"

            with open(filepath, mode = "w", encoding = "utf-8") as f:
                f.writelines(text[:-1])