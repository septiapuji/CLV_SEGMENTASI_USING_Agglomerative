import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

pip install pandas

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from math import ceil

from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.manifold import TSNE
import plotly
plotly.offline.init_notebook_mode(connected=True)

from ipywidgets import interact, interactive, fixed, interact_manual,VBox,HBox,Layout
import ipywidgets as widgets
import datetime
from datetime import date
import scipy.cluster.hierarchy as sch

dat=pd.read_csv('/kaggle/input/kprifizzz/fixkpri.csv', encoding= 'unicode_escape')
dat.head()

data = dat.copy()

data['InvoiceDate']=pd.to_datetime(data['InvoiceDate'])
data['Sales'] = data.Quantity*data.UnitPrice
data['Year']=data.InvoiceDate.dt.year
data['Month']=data.InvoiceDate.dt.month
data['Week']=data.InvoiceDate.dt.week
#data.InvoiceDate.dt.week
#wk = dt.isocalendar()[1]
data['Year_Month']=data.InvoiceDate.dt.to_period('M')
data['Hour']=data.InvoiceDate.dt.hour
data['Day']=data.InvoiceDate.dt.day
data['is_cancelled']=data.InvoiceNo.apply(lambda x: 'Yes' if x[0]=='C' else 'No')
data['weekday'] = data.InvoiceDate.dt.day_name()
data['Quarter'] = data.Month.apply(lambda m:'Q'+str(ceil(m/4)))
data['Date']=pd.to_datetime(data[['Year','Month','Day']])
data.head(150)

df = px.data.gapminder()
df = df [['country','iso_alpha']]
data = pd.merge(data,df[['country','iso_alpha']],left_on='Country',right_on='country',how='left').drop(columns=['country'])
print(data)
# del df

data_=data[data.is_cancelled=='No']
del data

sales_by_date = data_.groupby(by='Date')['Sales'].sum().reset_index()
fig = go.Figure(data=go.Scatter(x=sales_by_date.Date,y=sales_by_date.Sales
                                ,line = dict(color='black', width=1.5)))
fig.update_layout(xaxis_title="Date",yaxis_title="Sales",title='Daily Sales',template='ggplot2')
fig.show()

customer_by_month1 = data_.groupby('CustomerId')['Date'].min().reset_index()
customer_by_month1['days'] = pd.TimedeltaIndex(customer_by_month1.Date.dt.day,unit="D")
customer_by_month1['Month'] = customer_by_month1.Date- customer_by_month1.days+pd.DateOffset(days=1)
customer_by_month1['Quarter_acquisition'] = customer_by_month1['Month'].dt.quarter.apply(lambda x:'Q'+str(x))
customer_by_month1['Year_acquisition'] = customer_by_month1['Month'].dt.year
customer_by_month = data_.groupby(by = customer_by_month1.Month)['CustomerId'].size().reset_index()
customer_by_month.sort_values(by ='Month',ascending=True,inplace=True)
customer_by_month['cum_customer'] = np.cumsum(customer_by_month.CustomerId)
customer_by_month['Month_1'] = customer_by_month['Month'].dt.strftime('%b-%y')

plt.style.use('ggplot')
plt.figure(figsize=(20,5))
plt.plot(customer_by_month.Month_1,customer_by_month.cum_customer,'bo-',color='black')

for d,c in zip(customer_by_month['Month_1'],customer_by_month['cum_customer']):

    label = "{:.0f}".format(c)

    plt.annotate(label, 
                 (d,c), 
                 textcoords="offset points"
                 , bbox=dict(boxstyle="round", fc="none", ec="gray")
                 #,arrowprops=dict(arrowstyle="-",connectionstyle="angle,angleA=0,angleB=10,rad=90")
                 ,xytext=(0,10),
                 ha='center') 
plt.show()

del customer_by_month

sales_by_hour = data_.groupby(by='Hour')['Sales'].sum().reset_index()
sales_by_weekday = data_.groupby(by='weekday')['Sales'].sum().reset_index()

fig = make_subplots(rows=1, cols=2,subplot_titles=("Total Hourly Sales", "Total Sales by Weekday"))
fig.add_trace(go.Bar(y=sales_by_hour.Hour, x=sales_by_hour.Sales,orientation='h'),row=1, col=1)
fig.add_trace(go.Bar(x=sales_by_weekday.weekday, y=sales_by_weekday.Sales),row=1, col=2)
fig.update_layout(height=700, width=800,template='ggplot2')
fig.update_xaxes(title_text="Sales", row=1, col=1)
fig.update_xaxes(title_text="Weekday", row=1, col=2)
fig.update_yaxes(title_text="Hours", row=1, col=1)
fig.update_yaxes(title_text="Sales", row=1, col=2)
fig.show()


del [sales_by_hour,sales_by_weekday]

LRFM = data_.groupby('CustomerId').agg(Frequency=pd.NamedAgg(column="InvoiceNo", aggfunc="nunique")
                                        ,Monetary=pd.NamedAgg(column="Sales", aggfunc="sum"),
                                         Recency = pd.NamedAgg(column='InvoiceDate',aggfunc='min')).reset_index()
length = data_.groupby('CustomerId')['Date'].max() - data_.groupby('CustomerId')['Date'].min()
length =  (length/np.timedelta64(1, 'D')).reset_index()
length.columns = ['CustomerId','Length_of_stay']

LRFM = LRFM.merge(length,on='CustomerId',how='inner')
del length

LRFM.head(100)

e = LRFM['Recency'].min()
print('minimun :'+ str(e))

LRFM['Recency'] = LRFM['Recency'].apply(lambda x : (x - e).days)

LRFM.head(10)

a=(LRFM['Frequency'].max())
b=(LRFM['Frequency'].min())
print(a,b)

c= (LRFM['Monetary'].max())
d= (LRFM['Monetary'].min())
print(c,d)

e= (LRFM['Recency'].max())
f= (LRFM['Recency'].min())
print(e,f)

g= (LRFM['Length_of_stay'].max())
h= (LRFM['Length_of_stay'].min())
print(g,h)


LRFM.head(15)

#urutan
LRFM_1= LRFM[['Length_of_stay',"Recency","Frequency","Monetary"]]
LRFM_1.head()

from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1)) #inisialisasi normalisasi MinMax
LRFM_2 = min_max_scaler.fit_transform(LRFM_1) #transformasi MinMax untuk fitur

print("dataset setelah dinormalisasi :")
print(LRFM_2)

LRFM_3= pd.DataFrame(LRFM_2)

print(LRFM_3)

df = pd.DataFrame(dat)
CustomerId = []
#df.insert(0, column="CustomerId", value=range(2002))
LRFM_3.columns = ["Length_of_stay","Recency","Frequency","Monetary"]

LRFM_3.head()

LRFM_3.to_csv('LRFM_3.csv',float_format='%.2f')

plt.figure(figsize=(10, 7))  
plt.title("Dendrograms")  
dend = sch.dendrogram(sch.linkage(LRFM_3, method='ward'))

from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 10, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(LRFM_3)

# Silhouette analysis
range_n_clusters = [2,3,4,6]

for num_clusters in range_n_clusters:
    
    hc = AgglomerativeClustering(n_clusters = num_clusters , affinity = 'euclidean', linkage = 'ward') #inisialisasi hc digunakan untuk mencari cluster berdasarkan num_clusters
    y_hc = hc.fit_predict(LRFM_3)
    cluster_labels = hc.labels_
    
    # silhouette score
    silhouette_avg = silhouette_score(LRFM_3, cluster_labels)
    print("For n_clusters={0}, the silhouette score is {1}".format(num_clusters, silhouette_avg))

X=LRFM_3
cluster_lbls = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward').fit_predict(X)
X['cluster'] = cluster_lbls
X['sample_silhouette_values'] = silhouette_samples(X, cluster_lbls)
X['txt']=X.cluster.apply(lambda x:'Cluster '+str(x))

df = X.groupby('cluster').agg({'cluster':'size', 'Monetary':'mean','Frequency':'mean','Recency':'mean','Length_of_stay':'mean'}) \
       .rename(columns={'cluster':'Size','Monetary':'Avg Sales','Frequency':'Avg Frequency','Recency':'Avg Recency','Length_of_stay':'Avg Lenght of Stay'}) \
       .reset_index().sort_values(by = 'Avg Sales')

cluster_map ={'Cluster 0':'yellow','Cluster 1':'purple'}

txt =['Size = {0:.0f}'.format(i) for i in df.Size]
df['cluster']=df.cluster.apply(lambda x:'Cluster '+str(x))
df['Group']=df.cluster.map(cluster_map)

fig = make_subplots(rows=1, cols=4,subplot_titles=("Avg Sales", "Avg Frequency",'Avg Recency','Avg Lenght of Stay'))

fig.add_trace(go.Bar(y=df.cluster, x=df['Avg Sales'],hovertext=txt
                        ,text=txt,textposition='auto',marker_color=df.Group,orientation='h'),row=1, col=1)
fig.add_trace(go.Bar(y=df.cluster, x=df['Avg Frequency'],hovertext=txt
                        ,text=txt,textposition='auto',marker_color=df.Group,orientation='h'),row=1, col=2)
fig.add_trace(go.Bar(y=df.cluster, x=df['Avg Recency'],hovertext=txt
                        ,text=txt,textposition='auto',marker_color=df.Group,orientation='h'),row=1, col=3)
fig.add_trace(go.Bar(y=df.cluster, x=df['Avg Lenght of Stay'],hovertext=txt
                        ,text=txt,textposition='auto',marker_color=df.Group,orientation='h'),row=1, col=4)

fig.update_traces(marker_line_color='rgb(8,48,107)', marker_line_width=1.5, opacity=0.8)
fig.update_layout(title_text='Cluster Size',width = 800,height=600,template='ggplot2'
                  ,font=dict(family="Courier New, monospace",size=10,color="RebeccaPurple"))

fig.show()

#figure LRF
#3 D view of clusters
fig = go.Figure(data=[go.Scatter3d(x=X.Length_of_stay,y=X.Recency,z=X.Frequency,mode='markers'
                                   ,marker=dict(size=4,color=X.cluster
                                                ,colorscale='Viridis',opacity=0.8))])

# tight layout
fig.update_layout(margin=dict(l=1, r=2, b=1, t=1)
                  ,scene=dict(xaxis=dict(title='Length_of_stay')
                              ,yaxis=dict(title='Recency')
                              ,zaxis=dict(title='Frequency')),width=700,height=500)
fig.show()

#figure LRM
#3 D view of clusters
fig = go.Figure(data=[go.Scatter3d(x=X.Length_of_stay,y=X.Recency,z=X.Monetary,mode='markers'
                                   ,marker=dict(size=4,color=X.cluster
                                                ,colorscale='Viridis',opacity=0.8))])

# tight layout
fig.update_layout(margin=dict(l=1, r=2, b=1, t=1)
                  ,scene=dict(xaxis=dict(title='Length_of_stay')
                              ,yaxis=dict(title='Recency')
                              ,zaxis=dict(title='Monetary')),width=700,height=500)
fig.show()

#figure LFM
#3 D view of clusters
fig = go.Figure(data=[go.Scatter3d(x=X.Length_of_stay,y=X.Frequency,z=X.Monetary,mode='markers'
                                   ,marker=dict(size=4,color=X.cluster
                                                ,colorscale='Viridis',opacity=0.8))])

# tight layout
fig.update_layout(margin=dict(l=1, r=2, b=1, t=1)
                  ,scene=dict(xaxis=dict(title='X.Length_of_stay')
                              ,yaxis=dict(title='Frequency')
                              ,zaxis=dict(title='Monetary')),width=700,height=500)
fig.show()

#figure RFM
#3 D view of clusters
fig = go.Figure(data=[go.Scatter3d(x=X.Recency,y=X.Frequency,z=X.Monetary,mode='markers'
                                   ,marker=dict(size=4,color=X.cluster
                                                ,colorscale='Viridis',opacity=0.8))])

# tight layout
fig.update_layout(margin=dict(l=1, r=2, b=1, t=1)
                  ,scene=dict(xaxis=dict(title='Recency')
                              ,yaxis=dict(title='Frequency')
                              ,zaxis=dict(title='Monetary')),width=700,height=500)
fig.show()

#figure LR
#2 D view of clusters
fig = go.Figure(data=[go.Scatter(x=X.Length_of_stay,y=X.Recency,mode='markers'
                                   ,marker=dict(size=4,color=X.cluster
                                                ,colorscale='Viridis',opacity=0.8))])

# tight layout
fig.update_layout(margin=dict(l=1, r=2, b=1, t=1)
                  ,scene=dict(xaxis=dict(title='Length_of_stay')
                              ,yaxis=dict(title='Recency')),width=700,height=500)
fig.show()

#figure LF
#2 D view of clusters
fig = go.Figure(data=[go.Scatter(x=X.Length_of_stay,y=X.Frequency,mode='markers'
                                   ,marker=dict(size=4,color=X.cluster
                                                ,colorscale='Viridis',opacity=0.8))])

# tight layout
fig.update_layout(margin=dict(l=1, r=2, b=1, t=1)
                  ,scene=dict(xaxis=dict(title='Length_of_stay')
                              ,yaxis=dict(title='Frequency')),width=700,height=500)
fig.show()

#figure LM
#2 D view of clusters
fig = go.Figure(data=[go.Scatter(x=X.Length_of_stay,y=X.Monetary,mode='markers'
                                   ,marker=dict(size=4,color=X.cluster
                                                ,colorscale='Viridis',opacity=0.8))])

# tight layout
fig.update_layout(margin=dict(l=1, r=2, b=1, t=1)
                  ,scene=dict(xaxis=dict(title='Lenght_of_Stay')
                              ,yaxis=dict(title='Monetary')),width=700,height=500)
fig.show()

#figure RM
#2 D view of clusters
fig = go.Figure(data=[go.Scatter(x=X.Recency,y=X.Monetary,mode='markers'
                                   ,marker=dict(size=4,color=X.cluster
                                                ,colorscale='Viridis',opacity=0.8))])

# tight layout
fig.update_layout(margin=dict(l=1, r=2, b=1, t=1)
                  ,scene=dict(xaxis=dict(title='Recency')
                              ,yaxis=dict(title='Monetary')),width=700,height=500)
fig.show()

#figure RM
#2 D view of clusters
fig = go.Figure(data=[go.Scatter(x=X.Frequency,y=X.Monetary,mode='markers'
                                   ,marker=dict(size=4,color=X.cluster
                                                ,colorscale='Viridis',opacity=0.8))])

# tight layout
fig.update_layout(margin=dict(l=1, r=2, b=1, t=1)
                  ,scene=dict(xaxis=dict(title='Frequency')
                              ,yaxis=dict(title='Monetary')),width=700,height=500)
fig.show()

#figure RF
#2 D view of clusters
fig = go.Figure(data=[go.Scatter(x=X.Recency,y=X.Frequency,mode='markers'
                                   ,marker=dict(size=4,color=X.cluster
                                                ,colorscale='Viridis',opacity=0.8))])

# tight layout
fig.update_layout(margin=dict(l=1, r=2, b=1, t=1)
                  ,scene=dict(xaxis=dict(title='Recency')
                              ,yaxis=dict(title='Frequency')),width=700,height=500)
fig.show()


table = pd.DataFrame (df)

print(table)

#fddd =pd.DataFrame({'Avg Sales','Avg Recency', 'Avg Recency', 'Avg Lenght of Stay')
sk = table[['Avg Sales','Avg Frequency', 'Avg Recency', 'Avg Lenght of Stay']]

sk.values

# PERKALIAN CLV
#PR OBJECT TO ARRAY
blok1 = sk.values
blok2 = [0.238,0.088,0.326,0.348] #urutannya dibalik
nilaiclv1 = []
print (blok1.dtype)

for x in blok1:
    nilaiclv =[]
    for z,i in enumerate(x): 
        nilaiclv.append(i*blok2[z])
    nilaiclv1.append(nilaiclv)
print (nilaiclv1)

nilaiclvdeal = pd.DataFrame(nilaiclv1)

print(nilaiclvdeal)
