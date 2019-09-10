
import argparse
import os
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

from azureml.core import Run 
from utils import load_data

parser = argparse.ArgumentParser()
# parser.add_argument('--data-folder', type=str, dest='data_folder', help='data folder mounting point')
parser.add_argument('--property', type=str, dest='property', help='property mounting point')
parser.add_argument('--regularization', type=float, dest='reg', default=0.01, help='regulation rate')
args = parser.parse_args()

property = args.property
print('property:', property)