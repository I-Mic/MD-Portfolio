class TransactionDataset:
    """
    This class represents a transactional dataset, which consists of a list of transactions, where each
    transaction is a set of items. It provides methods to add transactions and to access the transactions.
    """
    def __init__(self,transactions = None):
        """
        Initializes a transaction dataset.
        """
        if transactions==None:
            self.transactions = []
        
        else:
            self.transactions = transactions
        
    def add_transaction(self, transaction):
        """
        Adds a transaction (as a set of items) to the dataset.

        Parameters:
        - transaction (set): The transaction to add.
        """
        self.transactions.append(transaction)

        
    def get_transactions(self):
        """
        Returns the list of transactions in the dataset.

        Returns:
        - list: The list of transactions.
        """
        return self.transactions
    
    def get_transaction_count(self):
        """
        Returns the number of transactions in the dataset.

        Returns:
        - int: The number of transactions.
        """
        return len(self.transactions)
    
    def get_itemset_count(self):
        """
        Returns the number of unique items in the dataset.

        Returns:
        - int: The number of unique items.
        """
        # Initialize an empty set to store unique items
        itemset = set()

        # Iterate over each transaction in the dataset
        for transaction in self.transactions:
            # Iterate over each item in the transaction
            for item in transaction:
                 # Add the item to the itemset set
                itemset.add(item)
        # Return the count of unique items in the itemset set
        return len(itemset)
    
    def get_frequent_itemsets(self):
        """
        Returns A dictionary containing all the frequent itemsets in the dataset and their respective counts.
        """
        # Initialize an empty dictionary to store itemsets and their counts
        count = {}

        # Iterate over each transaction in the dataset
        for transaction in self.transactions:
            # Iterate over each item in the transaction
            for item in transaction:
                # Check if the item is already in the count dictionary
                if item in count:
                    # If the item exists, increment its count by 1
                    count[item] += 1
                else:
                    # If the item does not exist, add it to the count dictionary
                    count[item] = 1
        
        # Sort the itemsets based on their counts in descending order
        frequent_items = sorted(count.items(), key=lambda x: x[1], reverse=True)
        
        # Return the dictionary of frequent itemsets and their counts
        return frequent_items

