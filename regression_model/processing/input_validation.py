import numpy as np
import re

def cast_num(num):
    """Takes input number and cast it into a float or a numpy null value"""
    try:
        num = float(num)
    except:
        num = np.nan
    return num

# def clean_text(text):
#     """Takes input text and extract numbers and spaces"""
#     text = re.sub(r'(\d+)','', text)  
#     text = re.sub(r'(\s+)','', text) 
#     return text