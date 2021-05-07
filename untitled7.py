# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 11:50:14 2021

@author: mdevasish
"""

def newPassword(a, b):
    # Write your code here
    min_len = min(len(a),len(b))
    print('length of a', len(a))
    print('length of b', len(b))
    hold = ''  
    hold_1 = ''
    hold_2 = '' 
    if min_len % 2 == 0:
        iter_min_len = int(min_len/2)
    else:
        iter_min_len = int((min_len+1)/2)
    print(iter_min_len)         
    if len(a) == len(b):
        for i in range(iter_min_len):
            output_1 = a[i]+b[i]
            hold_1 = hold_1+output_1
            if i != iter_min_len-1:
                output_2 = a[min_len-i-1]+b[min_len-i-1]
                hold_2 = output_2+hold_2
                print(hold_1,hold_2)
        return hold_1+hold_2
    for i in range(iter_min_len):
        output_1 = a[i]+b[i]
        hold_1 = hold_1+output_1
        if i != iter_min_len-1:
            output_2 = a[min_len-i-1]+b[min_len-i-1]
            hold_2 = output_2+hold_2
    hold = hold_1 + hold_2
    if min_len == len(b):
        hold = hold+a[min_len:]
    else:
        hold = hold+b[min_len:]
    return hold




# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 11:18:00 2021

@author: mdevasish
"""

#!/bin/python3

import math
import os
import random
import re
import sys


#
# Complete the 'newPassword' function below.
#
# The function is expected to return a STRING.
# The function accepts following parameters:
#  1. STRING a
#  2. STRING b
#

def newPassword(a, b):
    # Write your code here
    min_len = min(len(a),len(b))
    print('length of a', len(a))
    print('length of b', len(b))
    hold = ''            
    if len(a) == len(b):
        for i in range(min_len/2):
            output_1 = a[i]+b[i]
            output_2 = a[min_len-i-1]+b[min_len-i-1]
            hold = hold+output
            print(hold)
        return hold
    for i in range(min_len):
        output = a[i]+b[i]
        hold = hold+output
        print(hold)
    if min_len == len(b):
        hold = hold+a[min_len:]
    else:
        hold = hold+b[min_len:]
    return hold
        
            
if __name__ == '__main__':