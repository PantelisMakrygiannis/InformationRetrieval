import os

def read_documents_from_directory(directory_path):
    documents = []
    for filename in os.listdir(directory_path):
        with open(os.path.join(directory_path, filename), 'r', encoding="utf8") as file:
            text = file.read()
            document_id = len(documents) + 1  # Εκχώρηση ενός μοναδικού ID με βάση τη θέση
            document_tuple = (document_id, text)
            documents.append(document_tuple)
    return documents

# Διαδρομή των αρχείων
docs_directory = '/content/docs'
queries_file = '/content/drive/MyDrive/Queries_20'

# Διάβασμα των documents και queries
documents = read_documents_from_directory(docs_directory)

# Διάβασμα των queries από τον φάκελο
def read_queries_from_file(file_path):
    queries = []
    with open(file_path, 'r', encoding="utf8") as file:
        text = file.read()
        queries_list = text.split('\n')
        queries_list = [query.strip() for query in queries_list if query.strip()]
        queries.extend(queries_list)
    return queries
# Read queries from file
queries = read_queries_from_file(queries_file)

# Ρύθμιση των απαιτήσεων του ColBERT 
nbits = 2
doc_maxlen = 300
max_id = 1209
index_name = 'index'
checkpoint = '/content/drive/MyDrive/colbertv2.0'

# Indexing
with Run().context(RunConfig(nranks=1, experiment='indexing')):
    config = ColBERTConfig(doc_maxlen=doc_maxlen, nbits=nbits, kmeans_niters=4)
    indexer = Indexer(checkpoint=checkpoint, config=config)
    indexer.index(name=index_name, collection=[doc[1] for doc in documents[:max_id]], overwrite=True)

# Searching
with Run().context(RunConfig(experiment='indexing')):
    searcher = Searcher(index=index_name, collection=[doc[1] for doc in documents])


# Επανάληψη για όλα τα queries
for query in queries:
    print(f"#> {query}")

    # Εύρεση των top-1209 passages για κάθε query
    results = searcher.search(query, k=1209)
# Επανάληψη για όλα τα queries
for query in queries:
    print(f"#> {query}")

    # Εύρεση των passages για κάθε query
    results = searcher.search(query, k=1209)  # Ανάκτση όλων των passages

    # Εκτύπωση των passages που ανακτήθηκαν με σχετικό score
    for i, (passage_id, passage_rank, passage_score) in enumerate(zip(*results)):
        document_id = documents[passage_id][0]
        print(f"\t [{document_id}] \t\t {passage_score:.1f} \t\t {passage_id}")


import csv

# Δημιουργία dictionary για αποθήκευση λιστών των document IDs για κάθε query
query_document_lists = {}


# Επανάληψη για όλα τα queries
for query in queries:
    print(f"#> {query}")

    # Εύρεση των passages για κάθε query
    results = searcher.search(query, k=1209)  # Ανάκτση όλων των passages

    # Προσθήκη των αποτελεσμάτων στην λίστα
    result_list = []
    for i, (passage_id, passage_rank, passage_score) in enumerate(zip(*results)):
        result_list.append(passage_id)

    # Καταχώρηση της λίστας για το τέχον querie στο dictionary
    query_document_lists[query] = result_list

# Εκτύπωση του dictionary
print(query_document_lists)

# Αποθήκευση του dictionary σε ένα CSV έγγραφο
csv_file_path = '/content/colbert_result.csv'
with open(csv_file_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    # Εγγραφή του header στο αρχείο
    writer.writerow(['Query', 'DocumentIDs'])

    # Εγγραφή των data στο αρχείο
    for query, document_list in query_document_lists.items():
        writer.writerow([query, document_list])

print(f"Query document lists saved to {csv_file_path}")