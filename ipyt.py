# coding: utf-8
import numpy as np
import datetime as dt
import calendar as cd
y = dt.datetime.now().year
y
m = dt.datetime.now().month
m
days = cd.monthrange(y,m+1)[1]
days
arMonth = np.zeros((days,),dtype=int)
arMonth
ind = np.random.randint(0,days+1,7)
ind
