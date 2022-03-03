import numpy as np

def cast_num(num):
    """Takes input number and cast it into a float or a numpy null value"""
    try:
        num = float(num)
    except:
        num = np.nan
    return num
