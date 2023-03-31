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
        itemset = set()
        for transaction in self.transactions:
            for item in transaction:
                itemset.add(item)
        return len(itemset)
    
    def get_frequent_itemsets(self):
        item_counts = {}
        for transaction in self.transactions:
            for item in transaction:
                if item in item_counts:
                    item_counts[item] += 1
                else:
                    item_counts[item] = 1
        freq_items = sorted(item_counts.items(), key=lambda x: x[1], reverse=True)
        return freq_items

