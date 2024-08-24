import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc

# Δημιουργία dictionary για αποθήκευση των relevant documents για κάθε query
relevant_docs_dict = {}

# Διαδρομή του αρχείου που περιέχει τα δεδομένα του colBERT
colbert_path = '/../colbert_result.csv'
relevant_documents_path = '/../Relevant_20'

# Συμπλήρωση του relevant_docs_dict με relevant documents για κάθε query
with open(relevant_documents_path, 'r', encoding='utf-8') as relevant_file:
    for query_id, line in enumerate(relevant_file, start=1):
        parts = line.strip().split()
        if not parts:
            continue  # Παράληψη κενών γραμμών
        relevant_docs = set(map(int, parts))  # Μετατροπή των relevant document IDs σε ακεραίους
        relevant_docs_dict[query_id] = relevant_docs
    

with open(colbert_path, 'r', encoding='utf-8') as colbert_file:
    for query_id, line in enumerate(colbert_file, start=1):
        parts = line.strip().split(',')
        if not parts:
            continue  # Παράληψη κενών γραμμών
        colbert_docs = [int(part) for part in parts]  # Μετατροπή των relevant document IDs σε ακεραίους
        print(f"query: {query_id}")
        print(colbert_docs)

# Δημιουργία λιστών για την αποθήκευση αποτελεσμάτων των precision και recall για κάθε query
precision_results = []
recall_results = []

# Επανάληψη μέσω των queries στο relevant_docs_dict
for query_id, relevant_docs_set in relevant_docs_dict.items():
    
    # Υπολογισμός των precision και recall για διάφορα thresholds
    precisions = []
    recalls = []
    
    for threshold in range(1, len(colbert_docs) + 1):  # Χρήση όλων των ανακτημένων εγγράφων
        # Ανάκτηση του συνόλου των ανακτημένων εγγράφων με βάση το threshold
        retrieved_docs = set(map(int, colbert_docs[:threshold]))
        
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

