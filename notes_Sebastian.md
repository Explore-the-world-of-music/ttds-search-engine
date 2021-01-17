<ins>List of Features</ins>

Front end
-	Query completion (user might be able to select between different options)
-	Query correction (after misspelling)
-	Filter options for genre, time period, etc. (what ever we have in the data)
o	Filter the index or filter the search results?
-	Spotify/YouTube connection
-	Amazon reference to buy song/album

Search engine
-	Combine Boolean and ranked search (done)
-	Extend boolean queries for more than 2 terms (done)
-	Make query formulation consistent (open)
-	Use other tfidf parameters or even other ranking mechanism (own weighting)
-	Classification of theme of song (Genre is handled in Front end as it is directly given in the index, here use CW2)
-	Ranked results in case the user query is not accurate, meaning find similar queries and the results for that (e.g. I only love my bed and my mama I´m sorry… is replaced by I only love mahbed and my mama I´m sorry) . Added distance: If the word is not in index, find the closest match
-	System evaluation (based on CW2)

Recommendation engine (different interface than search engine or next step after a song and artist have been retrieved)
-	Similar artists or similar songs
-	Similar artists or similar songs in a different language (e.g. query in English and some results for similar German artists)
-	Topic search for a user to get new inspiration about songs he/she might like





<ins>IMPLEMENTATION NOTES</ins>

<ins>How to combine Boolean and ranked search?</ins>

For ranked queries each term in the query is checked for its (relative) frequency. For the different Boolean operators this is interpreted as follows:

-	AND: Count how often each term occurs only in the relevant documents.
-	AND NOT: Count how often t1 occurs only in the relevant documents. The occurrences for t2 are 0 for all relevant documents (as given by Boolean search)
-	OR: Count how often each term occurs only in the relevant documents
-	OR NOT: Count how often t1 occurs only in the relevant documents and multiply by 100, if t2 is not in the document “+1” is added
-	Phrase: How often is phrase in relevant documents
-	Proximity: How often is proximity successful in relevant documents

Important note with regards to CW1 ranked search:
The query is checked for a Boolean search component and if it is there, then the Boolean ranked search is performed. Otherwise ranked search as in CW1 is done

<ins>Query formulation or query induced search features</ins>

-	Use quotes to search for an exact phrase (done)
-	While searching for a phrase: use an asterisk within quotes to specify unknown or variable words (open).
-	Use the minus sign to eliminate results containing certain words, alternative to NOT (open)
-	When user just says “Christmas -Tree” we complement the query to “Christmas AND NOT TREE”
-	Filter search (to be implemented in the front end)
-	Suggest correction of misspelled words (to be implemented in the front end)


<ins>Extend boolean queries for more than 2 terms</ins>

-	Only Ands: As expected
-	Only ORs: As expected
-	A AND B OR C AND D: First AND and then OR and then AND. Interpret Boolean operators from left to right

Check if pattern is recognized correctly in execute search (especially for mixed phrase and logical operators)



