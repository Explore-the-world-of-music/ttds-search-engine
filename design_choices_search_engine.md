<ins>Consistent query formulation</ins>

-	Just words then ranked search will be triggered
-	Quotes to search for an exact phrase
-	“#(A,B,C)” for proximity search
-	Use an asterisk within quotes to specify unknown or variable words

Logical words are “AND, AND NOT, OR, OR NOT”. 
We use other syntax,
-	“--“ instead of NOT 
-	“&&” instead of AND
-	“||” instead of OR
-	“&&--” for AND NOT
-	“||--” for OR NOT
-	“!!word!!” for very important word

<ins>Extend Boolean queries for more than 2 terms</ins>

-	Only Ands: As expected
-	Only ORs: As expected
-	A AND B OR C AND D: First AND and then OR and then AND. Interpret Boolean operators from left to right and compare relevant documents stepwise
-	“A B C” AND D as expected
-	#15(A,B) AND E as expected

Please note that logical operators are read from left to right. This means that (A AND B) OR (C AND D) queries are interpreted as A AND B, OR C, AND D (step wise selection of relevant documents). Alternatively, the user can type in A AND B in the first query and then C AND D in the second query

<ins>Combination of Boolean and ranked search</ins>

The query is checked for a Boolean search component and if it is there, then the Boolean ranked search is performed. Otherwise ranked search as in CW1 is done

For ranked queries each term in the query is checked for its (relative) frequency. For the different Boolean operators this is interpreted as follows:

-	AND: Count how often each term occurs only in the relevant documents.
-	AND NOT: Count how often t1 occurs only in the relevant documents. The occurrences for t2 are 0 for all relevant documents (as given by Boolean search)
-	OR: Count how often each term occurs only in the relevant documents
-	OR NOT: Count how often t1 occurs only in the relevant documents, if t2 is not in the document “+0.5” is added
-	Phrase: How often is phrase in relevant documents
-	Proximity: How often is proximity successful in relevant documents

The scores are re-scaled in the end to make only small differences between documents visible



