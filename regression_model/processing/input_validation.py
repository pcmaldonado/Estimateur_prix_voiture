import numpy as np

def cast_num(num):
    """Takes input number and cast it into a float or a numpy nan value
    Arguments:
        num: str input number from user
    Returns:
        num: float number or np.nan"""
    try:
        num = float(num)
    except:
        num = np.nan
    return num
