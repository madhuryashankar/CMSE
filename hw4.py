import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Keep this to avoid unwanted warning on the wen app
st.set_option('deprecation.showPyplotGlobalUse', False)

# Loading dataset
data = pd.read_csv('https://raw.githubusercontent.com/madhuryashankar/CMSE/main/healthcare-dataset-stroke-data.csv')
X = data.drop('stroke',axis =1)
y = data['stroke']
print(data.columns.tolist())
# Title 
st.title('Stroke Prediction Dataset Exploration')
st.write('Interactive plots on Data')

# Sidebar
st.sidebar.title('Control Panel')

# Interactive Plots
st.sidebar.subheader('Select a feature for the histogram:')
selected_feature = st.sidebar.selectbox('Select a feature', X.columns.tolist())
bin_count = st.sidebar.slider('Number of Bins', min_value=1, max_value=100, value=20)
st.subheader(f'Histogram of {selected_feature}')
sns.histplot(data= data, x= data[selected_feature], hue='stroke', kde=True, bins=bin_count)
st.pyplot()

st.write(pd.DataFrame(data, columns=data.columns))

st.write('''
This Streamlit app allows the user to explore the stroke prediction dataset using interactive histogram plot.
You can choose the type of plot and features to visualize in the sidebar on the left.
You can also adjust the number of bins for histogram plot.
''')
