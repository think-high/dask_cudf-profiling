# -*- coding: utf-8 -*-
"""Common parts to all other modules, mainly utility functions.
"""
import pandas as pd
import cudf
import dask_cudf
import time

#dask_cudf profiling edit. 
#Adding verbose call
verbose = True

TYPE_CAT = 'CAT'
"""String: A categorical variable"""

TYPE_BOOL = 'BOOL'
"""String: A boolean variable"""

TYPE_NUM = 'NUM'
"""String: A numerical variable"""

TYPE_DATE = 'DATE'
"""String: A numeric variable"""

S_TYPE_CONST = 'CONST'
"""String: A constant variable"""

S_TYPE_UNIQUE = 'UNIQUE'
"""String: A unique variable"""

S_TYPE_UNSUPPORTED = 'UNSUPPORTED'
"""String: An unsupported variable"""

_VALUE_COUNTS_MEMO = {}

#dask_cudf profiling
#calculating distinct count once and then passing where required.
#Removing distinct_count calculation from here  
def get_groupby_statistic(data):
    """Calculate value counts and distinct count of a variable (technically a Series).

    The result is cached by column name in a global variable to avoid recomputing.

    Parameters
    ----------
    data : Series
        The data type of the Series.

    Returns
    -------
    list
        value count and distinct count
    """
    if data._name is not None and data._name in _VALUE_COUNTS_MEMO:
        return _VALUE_COUNTS_MEMO[data._name]

    #dask_cudf profiling
    #we don't need value_counts_with_nan. Commenting it
    #dask_cudf profiling timing
    # start = time.time()
    # value_counts_with_nan = data.value_counts()
    # end = time.time()

    # #time-profiling
    # if verbose:
    #     print("Total time elapsed in computing value_counts() is ", end-start)

    #dask_cudf profiling timing
    start = time.time()
    # TODO: Solve the dropna argument run issue 
    #value_counts_without_nan = data.value_counts(dropna=True)
    value_counts_without_nan = data.dropna().value_counts()
    end = time.time()

    #time-profiling
    if verbose:
        print("Total time elapsed in computing dropna + value_counts() is ", end-start)

    #dask_cudf profiling timing
    #removing distinct_count from here.
    # start = time.time()
    # distinct_count_with_nan = value_counts_with_nan.count()
    # end = time.time()

    # #time-profiling
    # if verbose:
    #     print("Total time elapsed in getting count() is ", end-start)

    #dask_cudf profiling timing
    #start = time.time()

    #Rahul dask_cudf profiling edit
    #Ignoring the check of "mixed" type for dask_cudf.Series. Will have to raise a Feature request
    # if not isinstance(value_counts_without_nan,dask_cudf.Series):
    #     if value_counts_without_nan.index.head(50).inferred_type == "mixed":
    #         raise TypeError('Not supported mixed type')

    #result = [value_counts_without_nan, distinct_count_with_nan]
    result = value_counts_without_nan

    #dask_cudf profiling: NEED ATTENTION
    #This call might break after this removal of distinct count 
    if data._name is not None:
        _VALUE_COUNTS_MEMO[data._name] = result
    #end = time.time()

    # #time-profiling
    # if verbose:
    #     print("Total time elapsed in end chunck of get_groupby_statistic is ", end-start)
    return result

#Depreciated old function
# def get_groupby_statistic(data):
#     """Calculate value counts and distinct count of a variable (technically a Series).

#     The result is cached by column name in a global variable to avoid recomputing.

#     Parameters
#     ----------
#     data : Series
#         The data type of the Series.

#     Returns
#     -------
#     list
#         value count and distinct count
#     """
#     if data._name is not None and data._name in _VALUE_COUNTS_MEMO:
#         return _VALUE_COUNTS_MEMO[data._name]

#     #dask_cudf profiling timing
#     start = time.time()
#     value_counts_with_nan = data.value_counts()
#     end = time.time()

#     #time-profiling
#     if verbose:
#         print("Total time elapsed in computing value_counts() is ", end-start)

#     #dask_cudf profiling timing
#     start = time.time()
#     value_counts_without_nan = data.dropna().value_counts()
#     end = time.time()

#     #time-profiling
#     if verbose:
#         print("Total time elapsed in computing dropna + value_counts() is ", end-start)

#     #dask_cudf profiling timing
#     start = time.time()
#     distinct_count_with_nan = value_counts_with_nan.count()
#     end = time.time()

#     #time-profiling
#     if verbose:
#         print("Total time elapsed in getting count() is ", end-start)

#     #dask_cudf profiling timing
#     start = time.time()

#     #Rahul dask_cudf profiling edit
#     #Ignoring the check of "mixed" type for dask_cudf.Series. Will have to raise a Feature request
#     if not isinstance(value_counts_without_nan,dask_cudf.Series):
#         if value_counts_without_nan.index.head(50).inferred_type == "mixed":
#             raise TypeError('Not supported mixed type')

#     result = [value_counts_without_nan, distinct_count_with_nan]

#     if data._name is not None:
#         _VALUE_COUNTS_MEMO[data._name] = result
#     end = time.time()

#     #time-profiling
#     if verbose:
#         print("Total time elapsed in end chunck of get_groupby_statistic is ", end-start)
#     return result


# TODO: Speed this up, it's too slow

#dask_cudf profiling
#calculating distinct count once and then passing where required.
_MEMO = {}
def get_vartype(data, distinct_count):
    """Infer the type of a variable (technically a Series).

    The types supported are split in standard types and special types.

    Standard types:
        * Categorical (`TYPE_CAT`): the default type if no other one can be determined
        * Numerical (`TYPE_NUM`): if it contains numbers
        * Boolean (`TYPE_BOOL`): at this time only detected if it contains boolean values, see todo
        * Date (`TYPE_DATE`): if it contains datetime

    Special types:
        * Constant (`S_TYPE_CONST`): if all values in the variable are equal
        * Unique (`S_TYPE_UNIQUE`): if all values in the variable are different
        * Unsupported (`S_TYPE_UNSUPPORTED`): if the variable is unsupported

     The result is cached by column name in a global variable to avoid recomputing.

    Parameters
    ----------
    data : Series
        The data type of the Series.

    Returns
    -------
    str
        The data type of the Series.

    Notes
    ----
        * Should improve verification when a categorical or numeric field has 3 values, it could be a categorical field
        or just a boolean with NaN values
        * #72: Numeric with low Distinct count should be treated as "Categorical"
    """
    if verbose:
        print("DataType at get_vartype is ", type(data))

    if data._name is not None and data._name in _MEMO:
        return _MEMO[data._name]

    vartype = None
    try:
        #dash_cudf profiling edit
        #Optimizing distinct count. Just using nunique() instead of groupby_statistic 
        # start = time.time()
        # distinct_count = 100 #data.nunique().compute()
        # end = time.time()
        
        # #time-profiling
        # if verbose:
        #     print("Total time elapsed in computing distinct count in get_vartype() ", end-start)


        #distinct_count = get_groupby_statistic(data)[1].compute()
        
        #dask_cudf profiling edit.
        #using len() instead of size for cudf.Series
        start = time.time()
        if isinstance(data,cudf.Series):
            leng = len(data)
        else:
            leng = data.size
        end = time.time()
        #time-profiling
        if verbose:
            print("Total time elapsed in getting size of the series ", end-start)


        start = time.time()
        if distinct_count <= 1:
            vartype = S_TYPE_CONST
        elif pd.api.types.is_bool_dtype(data) or (distinct_count == 2 and pd.api.types.is_numeric_dtype(data)):
            vartype = TYPE_BOOL
        elif pd.api.types.is_numeric_dtype(data):
            vartype = TYPE_NUM
        elif pd.api.types.is_datetime64_dtype(data):
            vartype = TYPE_DATE
        elif distinct_count == leng:
            vartype = S_TYPE_UNIQUE
        else:
            vartype = TYPE_CAT
        #dask_cudf profiling edit.
        end = time.time()
        
        #time-profiling
        if verbose:
            print("Total time elapsed in getting vartype of the series ", end-start)



    except:

        vartype = S_TYPE_UNSUPPORTED

    if data._name is not None:
        _MEMO[data._name] = vartype

    return vartype

def clear_cache():
    """Clear the cache stored as global variables"""
    global _MEMO, _VALUE_COUNTS_MEMO
    _MEMO = {}
    _VALUE_COUNTS_MEMO = {}
