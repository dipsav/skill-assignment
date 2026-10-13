"""filedict.py
a Persistent Dictionary in Python

Author: Erez Shinan
Date  : 31-May-2009

Fixed for current Python and xxhash (3 October 2026), nothing else changed:
  - MutableMapping is imported from collections.abc (collections.MutableMapping was removed in Python 3.10);
  - string keys are encoded as UTF-8 before hashing (xxhash 2.0+ refuses str; older versions hashed the UTF-8 bytes,
    so hash values -- and existing database files -- stay the same);
  - update() uses d.items() (d.iteritems() exists only in Python 2).
"""

# from collections import UserDict
import _pickle as pickle

try:
    from UserDict import UserDict
    from UserDict import DictMixin
except ImportError:
    from collections import UserDict
    try:
        from collections.abc import MutableMapping as DictMixin      # Python 3.3+ (required from 3.10)
    except ImportError:
        from collections import MutableMapping as DictMixin          # very old Python 3

import sqlite3

import hashlib, xxhash

def hash_funct(st):
    # return hashlib.md5(st.encode()).hexdigest()
    # return hash(st)
    # xxhash 2.0+ refuses str ("Strings must be encoded before hashing"); older versions hashed its UTF-8 bytes,
    # so encoding as UTF-8 here gives the same hash values and keeps existing database files readable.
    if isinstance(st, str):
        st = st.encode("utf-8")
    return xxhash.xxh32_intdigest(st)

class DefaultArg:
    pass

class Solutions:
    Sqlite3 = 0

class FileDict(DictMixin):
    "A dictionary that stores its data persistantly in a file"

    def __init__(self, solution=Solutions.Sqlite3, **options):
        assert solution == Solutions.Sqlite3
        try:
            self.__conn = options.pop('connection')
        except KeyError:
            filename = options.pop('filename')
            self.__conn = sqlite3.connect(filename)

        self.__tablename = options.pop('table', 'dict')

        self._nocommit = False

        assert not options, "Unrecognized options: %s" % options

        self.__conn.execute('create table if not exists %s (id integer primary key, hash integer, key blob, value blob);'%self.__tablename)
        self.__conn.execute('create index if not exists %s_index ON %s(hash);' % (self.__tablename, self.__tablename))
        self.__conn.commit()

    def _commit(self):
        if self._nocommit:
            return

        self.__conn.commit()

    def __pack(self, value):
        return sqlite3.Binary(pickle.dumps(value, -1))
    def __unpack(self, value):
        # return pickle.loads(str(value))
        return pickle.loads(value)

    def __get_id(self, key):
        key_hash = hash_funct(key)
        # print(key, key_hash)
        cursor = self.__conn.execute('select key,id from %s where hash=?;'%self.__tablename, (key_hash,))
        for k,id in cursor:
            if self.__unpack(k) == key:
                return id

        raise KeyError(key)

    def __getitem__(self, key):
        key_hash = hash_funct(key)
        cursor = self.__conn.execute('select key,value from %s where hash=?;'%self.__tablename, (key_hash,))
        for k,v in cursor:
            if self.__unpack(k) == key:
                return self.__unpack(v)

        raise KeyError(key)

    def __setitem(self, key, value):
        value_pickle = self.__pack(value)

        try:
            id = self.__get_id(key)
            cursor = self.__conn.execute('update %s set value=? where id=?;'%self.__tablename, (value_pickle, id) )
        except KeyError:
            key_pickle = self.__pack(key)
            key_hash = hash_funct(key)
            cursor = self.__conn.execute('insert into %s (hash, key, value) values (?, ?, ?);'
                    %self.__tablename, (key_hash, key_pickle, value_pickle) )

        assert cursor.rowcount == 1

    def __setitem__(self, key, value):
        self.__setitem(key, value)
        self._commit()

    def __delitem__(self, key):
        id = self.__get_id(key)
        cursor = self.__conn.execute('delete from %s where id=?;'%self.__tablename, (id,))
        if cursor.rowcount <= 0:
            raise KeyError(key)

        self._commit()

    def update(self, d):
        for k,v in d.items():                                             # was d.iteritems() (Python 2 only)
            self.__setitem(k, v)
        self._commit()

    def __iter__(self):
        return (self.__unpack(x[0]) for x in self.__conn.execute('select key from %s;'%self.__tablename) )
    def keys(self):
        return iter(self)
    def values(self):
        return (self.__unpack(x[0]) for x in self.__conn.execute('select value from %s;'%self.__tablename) )
    def items(self):
        return (map(self.__unpack, x) for x in self.__conn.execute('select key,value from %s;'%self.__tablename) )
    def iterkeys(self):
        return self.keys()
    def itervalues(self):
        return self.values()
    def iteritems(self):
        return self.items()

    def __contains__(self, key):
        try:
            self.__get_id(key)
            return True
        except KeyError:
            return False

    def __len__(self):
        return self.__conn.execute('select count(*) from %s;' % self.__tablename).fetchone()[0]

    def __del__(self):
        try:
            self.__conn
        except AttributeError:
            pass
        else:
            self.__conn.commit()

    @property
    def batch(self):
        return self._Batch(self)

    class _Batch:
        def __init__(self, d):
            self.__d = d

        def __enter__(self):
            self.__old_nocommit = self.__d._nocommit
            self.__d._nocommit = True
            return self.__d

        def __exit__(self, type, value, traceback):
            self.__d._nocommit = self.__old_nocommit
            self.__d._commit()
            return True