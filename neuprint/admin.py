"""
Administration utilities for managing a neuprint server.
Using these tools requires admin privileges on the neuPrintHttp server.
"""
from requests import HTTPError
from .client import inject_client

class Transaction:
    """
    For admins only.
    Used to batch a set of operations into a single database transaction.
    
    This class is implemented as a context manager.
    
    Example:
    
        .. code-block:: python
        
            with Transaction('hemibrain') as t:
                t.query("MATCH (n :Neuron {bodyId: 1047426385}) SET m.type=TuBu4)")
    """    
    @inject_client
    def __init__(self, dataset, *, client=None):
        """
        Transaction constructor.
        
        Args:
            dataset:
                Name of the dataset to use.  Required.
            
            client:
                Client object to use.
        """
        # This requirement isn't technically necessary,
        # but hopefully it avoids some confusing mistakes.
        if client.dataset and client.dataset != dataset:
            msg = ("The dataset you provided does not match the client's dataset.\n"
                   "To avoid confusion, provide a client whose dataset matches the transaction dataset.")
            raise RuntimeError(msg)
        
        assert dataset, \
            "Transactions require an an explicit dataset."
        self.dataset = dataset
        self.client = client
        self.transaction_id = None
        self.killed = False
    

    def query(self, cypher, format='pandas'):
        """
        Make a custom cypher query within the context
        of this transaction (allows writes).
        """
        if self.transaction_id is None:
            raise RuntimeError("no transaction was created")

        url = f"{self.client.server}/api/raw/cypher/transaction/{self.transaction_id}/cypher"
        return self.client._fetch_cypher(url, cypher, self.dataset, format)

    
    def kill(self):
        """
        Kills (rolls back) transaction.
        """
        if self.transaction_id is None:
            raise RuntimeError("no transaction was created")

        url = f"{self.client.server}/api/raw/cypher/transaction/{self.transaction_id}/kill"
        self.client._fetch_json(url, ispost=True)
        self.killed = True
        self.transaction_id = None


    def __enter__(self):
        self._start()
        return self
    

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            if not self.killed:
                self._commit()
        elif not self.killed and self.transaction_id is not None:
            self.kill()


    def _start(self):
        try:
            url = f"{self.client.server}/api/raw/cypher/transaction"
            result = self.client._fetch_json(url, json={"dataset": self.dataset}, ispost=True)
            self.transaction_id = result["transaction_id"]
        except HTTPError as ex:
            if ex.response.status_code == 401:
                raise RuntimeError("Transaction request was denied. "
                                   "Do you have admin privileges on the neuprintHttp server "
                                   f"({self.client.server})?")
            raise


    def _commit(self):
        if self.transaction_id is None:
            raise RuntimeError("no transaction was created")
        url = f"{self.client.server}/api/raw/cypher/transaction/{self.transaction_id}/commit"
        self.client._fetch_json(url, ispost=True)
        self.transaction_id = None
