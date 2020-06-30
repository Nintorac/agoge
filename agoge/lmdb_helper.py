from pathlib import Path
from tempfile import TemporaryFile
import torch

import lmdb


class LMDBDataset():
    """
    LMDB interface that expects str keys and dict values
    """

    def __init__(self, db_path, map_size=200*1e9, readonly=True):
        """
        db_path - path to lmdb folder
        map_size - max size of database, defaults to 200gb
        """

        db_path = Path(db_path).expanduser().resolve()
        self.db = lmdb.open(str(db_path), map_size=map_size, writemap=True, map_async=True, readonly=readonly)

        self.begin = self.db.begin

    @staticmethod
    def _pickle_dict(dict):
        with TemporaryFile('wb+') as f:
            torch.save(dict, f)
            f.seek(0)
            return f.read()

    @staticmethod
    def _unpickle_dict(bytes_like):
        with TemporaryFile('wb+') as f:
            f.write(bytes_like)
            f.seek(0)
            return torch.load(f)

    def put(self, key, value, tx=None):
        close=False
        if tx is None:
            close = True
            tx = self.db.begin(write=True)

        tx.put(key.encode('utf-8'), self._pickle_dict(value))

        if close:
            tx.commit()

    def keys(self):
        """
        Fetch all the keys in the db
        """

        with self.db.begin() as tx:
            cursor = tx.cursor()
            keys = [i.decode() for i in cursor.iternext(keys=True, values=False)]

        return keys

    def get(self, key, tx=None):
        """
        Get the value of a key from the database. 
        """
        close = False
        if tx is None:
            # open transaction if one doesnt exist
            close = True
            tx = self.db.begin()        
        
        if isinstance(key, list):
            # get all values as list
            value = [self.get(key, tx) for key in key]
            
        else:
            # get single value
            value = tx.get(key.encode())

            if value is None:
                raise ValueError('key not in db')

            value = self._unpickle_dict(value)

        # close transaction if neccesary
        if close:
            tx.commit()
        

        return value

    def sync(self):
        """
        manually sync the database to disk
        """

        self.db.sync()