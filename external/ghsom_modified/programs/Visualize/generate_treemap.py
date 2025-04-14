import plotly.express as px
import plotly.offline as of
import pandas as pd
import argparse
# import get_ghsom_dim

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from programs.data_processing import get_ghsom_dim

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--name', type=str, default = None)
parser.add_argument('--tau1', type=float, default = 0.1)
parser.add_argument('--tau2', type=float, default = 0.01)
parser.add_argument('--feature', type=str, default = 'mean')

# parser.add_argument('--index', type=str, default = None)
args = parser.parse_args()

prefix = args.name
t1 = args.tau1
t2 = args.tau2
feature = args.feature
file = f'{prefix}-{t1}-{t2}'

layers,max_layer,number_of_digits = get_ghsom_dim.layers(file)
pathlist = list()


df = pd.read_csv('./applications/%s-%s-%s/data/%s_with_clustered_label-%s-%s.csv' % (prefix,t1,t2, prefix,t1,t2), encoding='utf-8')
for i in range(1,max_layer+2):
    column_name = f'clusterL{i}'
    if column_name in df.columns:
        pathlist.append(column_name)

df = df.fillna('')

fig = px.treemap(df, path=pathlist,
          color = feature,
          color_continuous_scale = 'RdBu',
          branchvalues = 'remainder'
          )
of.plot(fig, filename=('./applications/%s/graphs/%s_map.html' % (file, prefix)))
fig.show()


