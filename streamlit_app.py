# _*_ coding: utf-8 _*_
"""
Created on Tue Oct 25 10:15:40 2022

@author: enzo magal
"""
#import libs
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from datetime import datetime
from functions import *


#create dataframes from the function 
train_data, test_data, lookId_data = load_data()

train_aug = train_data.dropna()

train_fill = train_data.fillna(method='ffill')

# data used as base for data augmentation
X_aug, y_aug = prepare_x_y(train_aug)

# data used directly in the training set
X_fill, y_fill = prepare_x_y(train_fill)

#build dashboard
add_sidebar = st.sidebar.selectbox('Data State', ('Data Discovering','Deep Learning'))

## Total picture
if add_sidebar == 'Data Discovering':
    st.header('Data Discovering')
    get_images(X_aug[:12], y_aug[:12], 6, shrinkage=0.1, fileName='./plots/data_disc.png')
    st.image('./plots/data_disc.png')

if add_sidebar == 'Deep Learning':
    st.header('Deep Learning')