import streamlit as st
import pandas as pd
import seaborn as sns

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('https://raw.githubusercontent.com/madhuryashankar/CMSE/main/healthcare-dataset-stroke-data.csv')  # Adjust the file name if needed
    return df

df = load_data()
st.title('Stroke Prediction Data Set Explorer')
df

# User input to select a column for plotting
selected_column = st.selectbox('Select a column for plotting', df.columns)

# Plot using Seaborn
st.subheader(f'KDE Plot for {selected_column}')
plot = sns.kdeplot(data = df, x = df[selected_column])
st.pyplot(fig=plot.get_figure())
