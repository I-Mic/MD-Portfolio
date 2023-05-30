from src.md.data.transaction_data import TransactionDataset
from itertools import combinations

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

        # Initialize an empty set to store the candidate itemsets
        candidates = set()

        # Iterate over each pair of frequent (k-1)-size itemsets
        for first_set in freq_itemsets:
            for second_set in freq_itemsets:
                # Check if the union of the two sets has the desired size k
                if len(first_set.union(second_set)) == k:
                    # Add the union of the two sets to the candidate itemsets
                    candidates.add(first_set.union(second_set))

        # Return the set of candidate k-size itemsets
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

        # Create an empty dictionary to store the counts of frequent itemsets
        counts = {}
        # Iterate over each transaction in the list of transactions
        for transaction in transactions:
            # Iterate over each itemset in the set of itemsets
            for itemset in itemsets:
                # Check if the itemset is a subset of the current transaction
                if itemset.issubset(transaction):
                    # If the itemset is already in the counts dictionary, increment its count by 1
                    if itemset in counts:
                        counts[itemset] += 1
                    # If the itemset is not in the counts dictionary, initialize its count to 1
                    else:
                        counts[itemset] = 1

        # Create a new dictionary to store the frequent itemsets that satisfy the minimum support constraint
        freq_itemsets = {itemset: count for itemset, count in counts.items() if count >= min_support}
        return freq_itemsets

    
    freq_itemsets = {}
    # Generate one-size-itemsets
    one_size_itemsets = {frozenset([item]): count for item, count in dataset.get_frequent_itemsets()}
    # Update the freq_itemsets dictionary with the one-size-itemsets
    freq_itemsets.update(one_size_itemsets)

    k = 2
    # Generate k-size-itemsets
    while True:
        # Generate candidate k-size itemsets
        k_size_itemsets = generate_candidates(freq_itemsets, k)
        if not k_size_itemsets:
            break
        # Get frequent k-size-itemsets
        k_size_freq_itemsets = get_frequent_itemsets(dataset.get_transactions(), k_size_itemsets, min_support)
        if not k_size_freq_itemsets:
            break
        # Update the freq_itemsets dictionary with the frequent k-size itemsets
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
    rules (dic): A dictionary containing all rules with their respective confidence.
    """

    # Initialize an empty list to store the generated association rules
    association_rules = []

    # Iterate over each itemset in the frequent_itemsets dictionary
    for itemset in frequent_itemsets:
        # Only consider itemsets with more than one item for generating rules
        if len(itemset) > 1:

            # Generate all possible combinations of items on the left side of the association rule
            for i in range(len(itemset)-1):
                for item in combinations(itemset,i+1):
                    # Create a frozenset for the left side of the rule
                    left = frozenset(item)
                    # Determine the remaining items for the right side of the rule
                    right = itemset - left
                    # confidence = support/item_count
                    confidence = frequent_itemsets[itemset] / frequent_itemsets[left]
                    # Check if the confidence meets the minimum threshold
                    if confidence >= min_confidence:
                        # Add the rule to the association_rules list
                        association_rules.append((left, right, confidence))
                

    return association_rules