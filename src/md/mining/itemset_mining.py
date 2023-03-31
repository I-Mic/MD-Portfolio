from src.md.data.transaction_data import TransactionDataset

def frequent_itemsets(dataset:TransactionDataset, min_support):
    """
    This function takes a TransactionDataset object and a minimum support value as input and generates all the frequent 
    itemsets in the dataset that satisfy the minimum support constraint as according to the Apriori Algorithm.

    Parameters:
    dataset (TransactionDataset): An object of class TransactionDataset containing the dataset.
    min_support (float): The minimum support value for an itemset to be considered frequent.

    Returns:
    freq_itemsets (dict): A dictionary containing all the frequent itemsets in the dataset and their respective counts.
    """

    def generate_candidates(freq_itemsets, k):
        """
        This function takes a dictionary of frequent k-1 size itemsets and an integer size k as input and generates all the candidate 
        k-size-itemsets.

        Parameters:
        freq_itemsets (dict): A dictionary containing the frequent k-1 size itemsets in the dataset.
        k (int): The size of the itemsets to generate.

        Returns:
        candidates (set): A set of all the candidate k-size-itemsets.
        """
        candidates = set()
        for first_set in freq_itemsets:
            for second_set in freq_itemsets:
                if len(first_set.union(second_set)) == k:
                    candidates.add(first_set.union(second_set))
        return candidates
    

    def get_frequent_itemsets(transactions, itemsets, min_support):
        """
        This function takes a list of transactions, a set of itemsets and a minimum support value as input and generates all 
        the frequent itemsets and their counts in the transactions that satisfy the minimum support constraint.

        Parameters:
        transactions (list): A list of all the transactions in the dataset.
        itemsets (set): A set of all the itemsets to be considered.
        min_support (float): The minimum support value for an itemset to be considered frequent.

        Returns:
        freq_itemsets (dict): A dictionary containing the frequent itemsets in the dataset and their respective counts.
        """
        counts = {}
        for transaction in transactions:
            for itemset in itemsets:
                if itemset.issubset(transaction):
                    if itemset in counts:
                        counts[itemset] += 1
                    else:
                        counts[itemset] = 1

        freq_itemsets = {itemset: count for itemset, count in counts.items() if count >= min_support}
        return freq_itemsets

    freq_itemsets = {}
    # Generate one-size-itemsets
    one_size_itemsets = {frozenset([item]): count for item, count in dataset.get_frequent_itemsets()}
    freq_itemsets.update(one_size_itemsets)

    k = 2
    # Generate k-size-itemsets
    while True:
        k_size_itemsets = generate_candidates(freq_itemsets, k)
        if not k_size_itemsets:
            break
        # Get frequent k-size-itemsets
        k_size_freq_itemsets = get_frequent_itemsets(dataset.get_transactions(), k_size_itemsets, min_support)
        if not k_size_freq_itemsets:
            break
        freq_itemsets.update(k_size_freq_itemsets)
        k += 1

    return freq_itemsets



def generate_rules(frequent_itemsets, min_confidence):
    """
    This function takes a dictionary of frequent itemsets and a minimum confidence value as input and generates all the 
    association rules that satisfy the minimum confidence constraint.

    Parameters:
    frequent_itemsets (dict): A dictionary containing all the frequent itemsets in the dataset and their respective counts.
    min_confidence (float): The minimum confidence value for a rule to be considered strong.

    Returns:
    rules (dic): A dictionary containing all the antecedent and consequent items with their respective confidence.
    """
   
    rules = []
    for itemset in frequent_itemsets.keys():
        if len(itemset) > 1:
            for item in itemset:
                antecedent = frozenset([item])
                consequent = itemset - antecedent
                confidence = frequent_itemsets[consequent] / frequent_itemsets[antecedent]
                if confidence >= min_confidence:
                    rules.append((antecedent, consequent, confidence))
    return rules