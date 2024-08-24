import os
from collections import defaultdict
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc

inverted_index = defaultdict(list)

folder_path = '/../docs'

for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)

    if os.path.isfile(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()

        words = text.split()

        for word in set(words):
            inverted_index[word].append((filename, words.count(word)))

# Υπολογισμός της συχνότητας εμφάνισης κάθε λέξης
word_counts = defaultdict(int)
for word, documents in inverted_index.items():
    word_counts[word] = len(documents)

print(inverted_index)

# Υπολογισμός του tf και αποθήκευση σε ένα dictionary
tf_values = {}
for word, occurrences in inverted_index.items():
    for document, count in occurrences:
        tf = 1 + math.log10(count)
        if document in tf_values:
            tf_values[document][word] = tf
        else:
            tf_values[document] = {word: tf}

# Δημιουργία του DataFrame για το tf
tf_df = pd.DataFrame.from_dict(tf_values, orient='index')
tf_df.fillna(0, inplace=True)

# Υπολογισμός του idf και αποθήκευση σε ένα dictionary
idf_values = {}
total_documents = len(tf_df)
for word in inverted_index.keys():
    df = word_counts[word]
    idf = math.log10(total_documents / df) if df > 0 else 0
    idf_values[word] = idf

# Δημιουργία του DataFrame για το idf
idf_df = pd.DataFrame.from_dict(idf_values, orient='index', columns=['IDF'])

# Εμφάνιση του DataFrame
print(idf_df)

# Βάρη (weights)
weight_values = {}

for document, word_tf_values in tf_values.items():
    weight_values[document] = {}
    for word, tf in word_tf_values.items():
        weight = tf * idf_values[word]
        weight_values[document][word] = weight

# Δημιουργία του DataFrame για τα βάρη (weights)
weight_df = pd.DataFrame.from_dict(weight_values, orient='index')

# Αντικατάσταση NaN τιμών με 0
weight_df.fillna(0, inplace=True)

# Εμφάνιση του DataFrame με τα βάρη
print(weight_df)

# Διαδρομή του αρχείου ερωτημάτων
query_file_path = '/../Queries_20'

# Διάβασμα του αρχείου ερωτημάτων
with open(query_file_path, 'r', encoding='utf-8') as query_file:
    queries_list = query_file.readlines()

# Αρχικοποίηση του DataFrame
query_df = pd.DataFrame(index=range(1, len(queries_list) + 1), columns=inverted_index.keys())
query_df = query_df.fillna(0)  # Αρχικά, όλα τα κελιά είναι 0

# Επεξεργασία των queries
for i, query in enumerate(queries_list, start=1):
    query_words = query.strip().split()  # Διαχωρισμός του query σε λέξεις
    for word in query_words:
        if word.upper() in query_df.columns:
            query_df.at[i, word.upper()] = 1  # Αν η λέξη υπάρχει στο query, θέτουμε το κελί σε 1

# Εμφάνιση του DataFrame
print(query_df)

# Ονομασία του αρχείου εξόδου
output_file_path = '/../QueryResults.txt'

# Αποθήκευση του DataFrame στο αρχείο κειμένου
with open(output_file_path, 'w', encoding='utf-8') as output_file:
    output_file.write(query_df.to_string())

# Υπολογισμός συνολικού αριθμού μηδενικών και μονάδων
total_zeros = query_df.eq(0).sum().sum()
total_ones = query_df.eq(1).sum().sum()

# Εμφάνιση των αποτελεσμάτων
print(f"Συνολικά 0: {total_zeros}")
print(f"Συνολικά 1: {total_ones}\n")

# Δημιουργία του DataFrame για το query_idf
query_idf_values = {}

for document, word_tf_values in query_df.iterrows():
    query_idf_values[document] = {}
    for word, tf in word_tf_values.items():
        query_idf_values[document][word] = tf * idf_values[word]  #όπου tf η συχνότητα του όρου στα query με βάση τον πίνακα

# Δημιουργία του DataFrame για το query_idf
query_idf_df = pd.DataFrame.from_dict(query_idf_values, orient='index')

# Αντικατάσταση NaN τιμών με 0
query_idf_df.fillna(0, inplace=True)

# Εμφάνιση του DataFrame με το query_idf
print(query_idf_df)

# Πολλαπλασιασμός των DataFrames
cosine_sim = query_idf_df.dot(weight_df.T)

# Ευκλείδειες νόρμες των γραμμών των DataFrames
query_idf_norm = np.linalg.norm(query_idf_df.values, axis=1)
weight_norm = np.linalg.norm(weight_df.values, axis=1)

# Διαίρεση με το γινόμενο των ευκλείδιων νορμών
cosine_sim = cosine_sim.divide(np.outer(query_idf_norm, weight_norm))

# Εμφάνιση του αποτελέσματος
print(cosine_sim)


# Δημιουργία dictionary για αποθήκευση των relevant documents για κάθε query
relevant_docs_dict = {}

# Διαδρομή του αρχείου 
relevant_documents_path = '/../Relevant_20'

# Συμπλήρωση του relevant_docs_dict με relevant documents για κάθε query
with open(relevant_documents_path, 'r', encoding='utf-8') as relevant_file:
    for query_id, line in enumerate(relevant_file, start=1):
        parts = line.strip().split()
        if not parts:
            continue  # Παράληψη κενών γραμμών
        relevant_docs = set(map(int, parts))  # Μετατροπή των relevant document IDs σε ακεραίους 
        relevant_docs_dict[query_id] = relevant_docs


# Επανάληψη μέσω των queries στο relevant_docs_dict
for query_id, relevant_docs_set in relevant_docs_dict.items():
    # Λήψη των ταξινομημένων τιμών και των αντίστοιχων κειμένων για το τρέχον query
    row = cosine_sim.loc[query_id]
    sorted_values = row.sort_values(ascending=False).values
    sorted_texts = row.sort_values(ascending=False).index
    
# Δημιουργία λιστών για την αποθήκευση αποτελεσμάτων των precision και recall για κάθε query
precision_results = []
recall_results = []

# Επανάληψη μέσω των queries στο relevant_docs_dict
for query_id, relevant_docs_set in relevant_docs_dict.items():
    # Λήψη των ταξινομημένων τιμών και των αντίστοιχων κειμένων για το τρέχον query
    row = cosine_sim.loc[query_id]
    sorted_values = row.sort_values(ascending=False).values
    sorted_texts = row.sort_values(ascending=False).index
    
    # Υπολογισμός των precision και recall για διάφορα thresholds
    precisions = []
    recalls = []
    
    for threshold in range(1, len(sorted_texts) + 1):  # Χρήση όλων των ανακτημένων εγγράφων 
        # Ανάκτηση του συνόλου των ανακτημένων εγγράφων με βάση το threshold
        retrieved_docs = set(map(int, sorted_texts[:threshold]))
        
        # Υπολογισμός precision and recall
        if len(relevant_docs_set) > 0:
            precision = len(relevant_docs_set.intersection(retrieved_docs)) / len(retrieved_docs)
        else:
            precision = 0
        
        recall = len(relevant_docs_set.intersection(retrieved_docs)) / len(relevant_docs_set) if len(relevant_docs_set) > 0 else 0
        
        # Προσθήκη των precision and recall στις λίστες
        precisions.append(precision)
        recalls.append(recall)
    
    # Προσθήκη των λιστών precision and recall στα αποτελέσματα
    precision_results.append(precisions)
    recall_results.append(recalls)

# Δημιουργία λίστας για την αποθήκευση του εμβαδού (AUC-PR) για κάθε query
auc_pr_results = []

# Επανάληψη μέσω των queries στο relevant_docs_dict
for query_id, (precision_values, recall_values) in enumerate(zip(precision_results, recall_results), start=1):

    # Υπολογισμός του εμβαδού κάτω την precision-recall καμπύλη
    auc_pr = auc(recall_values, precision_values)

    # Προσθήκη του AUC-PR στα αποτελέσματα
    auc_pr_results.append(auc_pr)

    # Δημιουργία γραφήματος για κάθε query
    plt.figure()
    plt.plot(recall_values, precision_values, marker='o')
    plt.title(f"Precision - Recall for Query {query_id}\nAUC-PR: {auc_pr:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(True)
    plt.show()

# Εκτύπωση των αποτελεσμάτων AUC-PR για κάθε query
for query_id, auc_pr in enumerate(auc_pr_results, start=1):
    print(f"Query {query_id}: AUC-PR = {auc_pr:.4f}")


# Υπολογισμός του Mean Average Precision (MAP)
average_precision_values = []

for precisions, relevant_docs_set in zip(precision_results, relevant_docs_dict.values()):
    # Υπολογισμός του AP για το query
    average_precision = 0
    for i, precision in enumerate(precisions):
        if i + 1 in relevant_docs_set:
            average_precision += precision

    average_precision /= len(relevant_docs_set)
    
    # Καταχώρηση της τιμής AP στη λίστα
    average_precision_values.append(average_precision)

# Υπολογισμός του Mean Average Precision (MAP)
map_value = np.mean(average_precision_values)

# Εκτύπωση της τιμής MAP 
print(f"Mean Average Precision (MAP): {map_value}")

# Σχεδίαση του MAP ως γραμμή
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(average_precision_values) + 1), average_precision_values, marker='o', linestyle='-')
plt.xlabel('Query')
plt.ylabel('Average Precision')
plt.title('Mean Average Precision (MAP)')
plt.show()


