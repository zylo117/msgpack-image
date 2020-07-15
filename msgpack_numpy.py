#!/usr/bin/env python

"""
Support for serialization of numpy data types with msgpack.
"""

# Copyright (c) 2013-2020, Lev E. Givon
# All rights reserved.
# Distributed under the terms of the BSD license:
# http://www.opensource.org/licenses/bsd-license

import functools
import sys

import cv2
import msgpack
import numpy as np
from msgpack import Packer as _Packer, Unpacker as _Unpacker, \
    unpack as _unpack, unpackb as _unpackb

ENABLE_LOSSY_CONVERSION = True
if ENABLE_LOSSY_CONVERSION:
    from rle_encoder.rle import rle_to_binary_mask_cython, binary_mask_to_rle_cython

    jpeg_compress_size_limit = 1024 ** 2
    is_boolmap = lambda data: isinstance(data, np.ndarray) and data.dtype == np.bool
    is_rle = lambda data: isinstance(data, dict) and 'size' in data and 'counts' in data

if sys.version_info >= (3, 0):
    if sys.platform == 'darwin':
        ndarray_to_bytes = lambda obj: obj.tobytes()
    else:
        ndarray_to_bytes = lambda obj: obj.data if obj.flags['C_CONTIGUOUS'] else obj.tobytes()

    num_to_bytes = lambda obj: obj.data


    def tostr(x):
        if isinstance(x, bytes):
            return x.decode()
        else:
            return str(x)
else:
    if sys.platform == 'darwin':
        ndarray_to_bytes = lambda obj: obj.tobytes()
    else:
        ndarray_to_bytes = lambda obj: memoryview(obj.data) if obj.flags['C_CONTIGUOUS'] else obj.tobytes()

    num_to_bytes = lambda obj: memoryview(obj.data)


    def tostr(x):
        return x


def encode(obj, chain=None):
    """
    Data encoder for serializing numpy data types.
    """
    if isinstance(obj, np.ndarray) and not is_boolmap(obj):
        # If the dtype is structured, store the interface description;
        # otherwise, store the corresponding array protocol type string:
        if obj.dtype.kind == 'V':
            kind = b'V'
            descr = obj.dtype.descr
        elif ENABLE_LOSSY_CONVERSION and len(obj.shape) == 3 and obj.shape[
            2] == 3 and obj.dtype == np.uint8 and sys.getsizeof(
            obj) > jpeg_compress_size_limit:
            kind = b'J'
            obj = cv2.imencode('.jpg', obj, [cv2.IMWRITE_JPEG_QUALITY, 80])[1]
            descr = obj.dtype.str
        else:
            kind = b''
            descr = obj.dtype.str
        return {b'nd': True,
                b'type': descr,
                b'kind': kind,
                b'shape': obj.shape,
                b'data': ndarray_to_bytes(obj)}
    elif isinstance(obj, (np.bool_, np.number)):
        return {b'nd': False,
                b'type': obj.dtype.str,
                b'data': num_to_bytes(obj)}
    elif isinstance(obj, complex):
        return {b'complex': True,
                b'data': obj.__repr__()}
    elif is_boolmap(obj):
        kind = b'R'
        descr = obj.dtype.str
        shape = obj.shape
        obj = binary_mask_to_rle_cython(obj)
        return {b'nd': True,
                b'type': descr,
                b'kind': kind,
                b'shape': shape,
                b'data': obj}
    else:
        return obj if chain is None else chain(obj)


def decode(obj, chain=None):
    """
    Decoder for deserializing numpy data types.
    """

    try:
        if b'nd' in obj:
            if obj[b'nd'] is True:

                # Check if b'kind' is in obj to enable decoding of data
                # serialized with older versions (#20):
                if b'kind' in obj and obj[b'kind'] == b'V':
                    descr = [tuple(tostr(t) if type(t) is bytes else t for t in d) \
                             for d in obj[b'type']]
                elif ENABLE_LOSSY_CONVERSION and b'kind' in obj and obj[b'kind'] == b'J':
                    descr = obj[b'type']
                    data = np.frombuffer(obj[b'data'], dtype=np.dtype(descr)).reshape(obj[b'shape'])
                    return cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
                elif ENABLE_LOSSY_CONVERSION and b'kind' in obj and obj[b'kind'] == b'R':
                    data = rle_to_binary_mask_cython(obj[b'data'])
                    return data
                else:
                    descr = obj[b'type']
                return np.frombuffer(obj[b'data'],
                                     dtype=np.dtype(descr)).reshape(obj[b'shape'])
            else:
                descr = obj[b'type']
                return np.frombuffer(obj[b'data'],
                                     dtype=np.dtype(descr))[0]
        elif b'complex' in obj:
            return complex(tostr(obj[b'data']))
        else:
            return obj if chain is None else chain(obj)
    except KeyError:
        return obj if chain is None else chain(obj)


# Maintain support for msgpack < 0.4.0:
if msgpack.version < (0, 4, 0):
    class Packer(_Packer):
        def __init__(self, default=None,
                     encoding='utf-8',
                     unicode_errors='strict',
                     use_single_float=False,
                     autoreset=1):
            default = functools.partial(encode, chain=default)
            super(Packer, self).__init__(default=default,
                                         encoding=encoding,
                                         unicode_errors=unicode_errors,
                                         use_single_float=use_single_float,
                                         autoreset=autoreset)


    class Unpacker(_Unpacker):
        def __init__(self, file_like=None, read_size=0, use_list=None,
                     object_hook=None,
                     object_pairs_hook=None, list_hook=None, encoding='utf-8',
                     unicode_errors='strict', max_buffer_size=0):
            object_hook = functools.partial(decode, chain=object_hook)
            super(Unpacker, self).__init__(file_like=file_like,
                                           read_size=read_size,
                                           use_list=use_list,
                                           object_hook=object_hook,
                                           object_pairs_hook=object_pairs_hook,
                                           list_hook=list_hook,
                                           encoding=encoding,
                                           unicode_errors=unicode_errors,
                                           max_buffer_size=max_buffer_size)

elif msgpack.version < (1, 0, 0):
    class Packer(_Packer):
        def __init__(self, default=None,
                     unicode_errors='strict',
                     use_single_float=False,
                     autoreset=1,
                     use_bin_type=True,
                     strict_types=False):
            default = functools.partial(encode, chain=default)
            super(Packer, self).__init__(default=default,
                                         unicode_errors=unicode_errors,
                                         use_single_float=use_single_float,
                                         autoreset=autoreset,
                                         use_bin_type=use_bin_type,
                                         strict_types=strict_types)


    class Unpacker(_Unpacker):
        def __init__(self, file_like=None, read_size=0, use_list=None,
                     raw=False,
                     object_hook=None,
                     object_pairs_hook=None, list_hook=None,
                     unicode_errors='strict', max_buffer_size=0,
                     ext_hook=msgpack.ExtType):
            object_hook = functools.partial(decode, chain=object_hook)
            super(Unpacker, self).__init__(file_like=file_like,
                                           read_size=read_size,
                                           use_list=use_list,
                                           raw=raw,
                                           object_hook=object_hook,
                                           object_pairs_hook=object_pairs_hook,
                                           list_hook=list_hook,
                                           unicode_errors=unicode_errors,
                                           max_buffer_size=max_buffer_size,
                                           ext_hook=ext_hook)

else:
    class Packer(_Packer):
        def __init__(self,
                     default=None,
                     use_single_float=False,
                     autoreset=True,
                     use_bin_type=True,
                     strict_types=False,
                     datetime=False,
                     unicode_errors=None):
            default = functools.partial(encode, chain=default)
            super(Packer, self).__init__(default=default,
                                         use_single_float=use_single_float,
                                         autoreset=autoreset,
                                         use_bin_type=use_bin_type,
                                         strict_types=strict_types,
                                         datetime=datetime,
                                         unicode_errors=unicode_errors)


    class Unpacker(_Unpacker):
        def __init__(self,
                     file_like=None,
                     read_size=0,
                     use_list=True,
                     raw=False,
                     timestamp=0,
                     strict_map_key=True,
                     object_hook=None,
                     object_pairs_hook=None,
                     list_hook=None,
                     unicode_errors=None,
                     max_buffer_size=100 * 1024 * 1024,
                     ext_hook=msgpack.ExtType,
                     max_str_len=-1,
                     max_bin_len=-1,
                     max_array_len=-1,
                     max_map_len=-1,
                     max_ext_len=-1):
            object_hook = functools.partial(decode, chain=object_hook)
            super(Unpacker, self).__init__(file_like=file_like,
                                           read_size=read_size,
                                           use_list=use_list,
                                           raw=raw,
                                           timestamp=timestamp,
                                           strict_map_key=strict_map_key,
                                           object_hook=object_hook,
                                           object_pairs_hook=object_pairs_hook,
                                           list_hook=list_hook,
                                           unicode_errors=unicode_errors,
                                           max_buffer_size=max_buffer_size,
                                           ext_hook=ext_hook,
                                           max_str_len=max_str_len,
                                           max_bin_len=max_bin_len,
                                           max_array_len=max_array_len,
                                           max_map_len=max_map_len,
                                           max_ext_len=max_ext_len)


def pack(o, stream, **kwargs):
    """
    Pack an object and write it to a stream.
    """

    packer = Packer(**kwargs)
    stream.write(packer.pack(o))


def packb(o, **kwargs):
    """
    Pack an object and return the packed bytes.
    """

    return Packer(**kwargs).pack(o)


def unpack(stream, **kwargs):
    """
    Unpack a packed object from a stream.
    """

    object_hook = kwargs.get('object_hook')
    kwargs['object_hook'] = functools.partial(decode, chain=object_hook)
    return _unpack(stream, **kwargs)


def unpackb(packed, **kwargs):
    """
    Unpack a packed object.
    """

    object_hook = kwargs.get('object_hook')
    kwargs['object_hook'] = functools.partial(decode, chain=object_hook)
    return _unpackb(packed, **kwargs)


load = unpack
loads = unpackb
dump = pack
dumps = packb


def patch():
    """
    Monkey patch msgpack module to enable support for serializing numpy types.
    """

    setattr(msgpack, 'Packer', Packer)
    setattr(msgpack, 'Unpacker', Unpacker)
    setattr(msgpack, 'load', unpack)
    setattr(msgpack, 'loads', unpackb)
    setattr(msgpack, 'dump', pack)
    setattr(msgpack, 'dumps', packb)
    setattr(msgpack, 'pack', pack)
    setattr(msgpack, 'packb', packb)
    setattr(msgpack, 'unpack', unpack)
    setattr(msgpack, 'unpackb', unpackb)
