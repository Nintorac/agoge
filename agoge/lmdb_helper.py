from pathlib import Path
from tempfile import TemporaryFile
import torch

import lmdb


class LMDBDataset():
    """
    LMDB interface that expects str keys and dict values
    """

    def __init__(self, db_path, map_size=200*1e9, readonly=True, max_dbs=0):
        """
        db_path - path to lmdb folder
        map_size - max size of database, defaults to 200gb
        """

        self.db_path = Path(db_path).expanduser().resolve()
        self.map_size = map_size
        self.readonly = readonly
        self.max_dbs = max_dbs

    @property
    def db(self):
        return lmdb.open(str(self.db_path), 
                map_size=self.map_size, 
                writemap=True, 
                map_async=True, 
                readonly=self.readonly,
                max_dbs=self.max_dbs
            )

    @property
    def begin(self):
        return self.db.begin

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

    def put(self, key, value, tx=None, db=None):
        """
        Put a new key, value pair in the database using transaction tx and
        into databae given in db

        key - string
        value - pickleable dictionary
        tx - lmdb.Transaction
        db - string
        """
        close=False
        if tx is None:
            env = self.db
            if db is not None:
                db = env.open_db(db.encode())
            close = True
            tx = env.begin(write=True, db=db)

        tx.put(key.encode('utf-8'), self._pickle_dict(value))

        if close:
            tx.commit()

    def keys(self, sort=True, db=None):
        """
        Fetch all the keys in the db
        """
        env = self.db
        if db is not None:
            db = env.open_db(db.encode())

        with env.begin(db=db) as tx:
            cursor = tx.cursor()
            keys = [i.decode() for i in cursor.iternext(keys=True, values=False)]
        if sort:
            keys.sort()
        return keys

    def get(self, key, tx=None, db=None):
        """
        Get the value of a key from the database.

        key - string
        tx - lmdb.Transaction
        db - string
        """
        close = False
        if tx is None:
            env = self.db
            if db is not None:
                db = env.open_db(db.encode())
            # open transaction if one doesnt exist
            close = True
            tx = env.begin(db=db)
        
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