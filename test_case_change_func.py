def find_common_dtype_and_compute(arr1, arr2, arr3):
    type1 = arr1.dtype
    type2 = arr2.dtype
    type3 = arr3.dtype
    common_type = np.promote_types(np.promote_types(type1, type2), type3)
    arr1_casted = arr1.astype(common_type)
    arr2_casted = arr2.astype(common_type)
    arr3_casted = arr3.astype(common_type)
    result = arr1_casted + arr2_casted - arr3_casted
    return result

import numpy as np 
def find_common_dtype_and_compute(arr1, arr2, arr3):
    type1 = arr1.dtype
    type2 = arr2.dtype
    type3 = arr3.dtype
    common_type = np.promote_types(np.promote_types(type1, type2), type3)
    arr1_casted = arr1.astype(common_type)
    arr2_casted = arr2.astype(common_type)
    arr3_casted = arr3.astype(common_type)
    result = arr1_casted + arr2_casted - arr3_casted
    return result
# Input data\ntest_data = [\n    (np.array([1, 2, 3], dtype=np.int32), np.array([4.0, 5.0, 6.0], dtype=np.float64), np.array([1, 2, 3], dtype=np.int32)),\n    (np.array([True, False, True], dtype=np.bool), np.array([1, 0, 1], dtype=np.int32), np.array([0, 1, 1], dtype=np.int32)),\n    (np.array([[1, 2], [3, 4]], dtype=np.int64), np.array([[5, 6], [7, 8]], dtype=np.float32), np.array([[9, 10], [11, 12]], dtype=np.float64))\n]\n\nfor arr1, arr2, arr3 in test_data:\n    try:\n        result = find_common_dtype_and_compute(arr1, arr2, arr3)\n        print(result)\n    except Exception as e:\n        print(\"error:\", e)\n```",



import numpy as np
def find_common_dtype_and_compute(arr1, arr2, arr3):
    type1 = arr1.dtype
    type2 = arr2.dtype
    type3 = arr3.dtype
    try:
        common_type = np.promote_types(np.promote_types(type1, type2), type3)
    except TypeError:
        common_type = np.result_type(type1, type2, type3)
        arr1_casted = arr1.astype(common_type)
        arr2_casted = arr2.astype(common_type)
        arr3_casted = arr3.astype(common_type)
        result = arr1_casted + arr2_casted - arr3_casted
        return result







def maximize_vertical_stack(arrays: list) -> int:
    import numpy as np
    stacked = np.vstack(arrays)
    max_sum = np.max(np.sum(stacked, axis=0))
    return max_sum