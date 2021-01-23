# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 00:24:01 2020

@author: JayNit
"""

import requests

url = 'http://localhost:5000/results'
r = requests.post(url,json={'REVIEW_TEXT':'Awesome food, Excellent Service'})

print(r.json())