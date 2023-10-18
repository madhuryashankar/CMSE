import streamlit as st
import plotly.express as px
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import hiplot as hip

# Keep this to avoid unwanted warning on the wen app
st.set_option('deprecation.showPyplotGlobalUse', False)

# Loading dataset
df = pd.read_csv('https://raw.githubusercontent.com/madhuryashankar/CMSE/main/healthcare-dataset-stroke-data.csv')

# Function to replace missing values with median
def replace_missing_with_median(df):
    df['bmi'] = df['bmi'].fillna(df['bmi'].median())
    # Add more columns as needed

# Function to filter data based on user selections
def filter_data(df, selected_work_type, selected_smoking_status, selected_age_range, selected_gender):
    filtered_data = df[(df['work_type'] == selected_work_type) &
                       (df['smoking_status'] == selected_smoking_status) &
                       (df['age'] >= selected_age_range[0]) & (df['age'] <= selected_age_range[1]) &
                       (df['gender'] == selected_gender)]
    return filtered_data

# Function to create bar plots of categorical features by diagnosis
def create_bar_plot(df, categorical_feature):
    fig = px.histogram(df, x="stroke", color=categorical_feature, barmode='group')
    return fig

# Function to create violin plots of numerical features by diagnosis
def create_violin_plot(df, numerical_feature):
    fig = px.violin(df, x="stroke", y=numerical_feature, box=True, hover_data=df.columns)
    return fig

# Function to create scatter plots with correlation analysis
def create_scatterplot_with_correlation(df, x_feature, y_feature, hue_feature):
    fig = px.scatter(df, x=x_feature, y=y_feature, color=hue_feature, trendline="ols")
    return fig

# Function to create a correlation matrix
def create_correlation_matrix(df, corr_range):
    numerical_features = df.select_dtypes(include=['float64']).columns
    selected_corr_data = df[numerical_features].corr()
    selected_corr_data = selected_corr_data[(selected_corr_data >= corr_range[0]) & (selected_corr_data <= corr_range[1])]
    return selected_corr_data

# Apply styling
st.set_page_config(
    page_title="Predicting Strokes: Insights from the Data",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)

# Set the title and description of the app
st.write('<h2 style="text-align:center; vertical-align:middle; line-height:2; color:#046366;">Predicting Strokes: Insights from the Data</h2>', unsafe_allow_html=True)

# Create tabs
tab1, tab2, tab3 = st.tabs(["About the Data", "Visualizations", "Playground"])

with tab1 :
    image_path = "bg.png"  # Replace with the actual file path

    # Check if the image file exists at the specified path
    try:
        with open(image_path, "rb") as image_file:
            img = Image.open(image_file)
            img = img.resize((img.width, 300))
            st.image(img, caption="Stroke Prediction", use_column_width=True)
    except FileNotFoundError:
        pass
  
    st.write("Stroke Prediction plays a pivotal role in predicting the likelihood of an individual experiencing a stroke. Strokes, as the second leading cause of death globally, accounting for approximately 11% of total deaths according to the World Health Organization (WHO), represent a critical healthcare challenge.")
    st.write("The 'Stroke Prediction Dataset' was sourced from Kaggle which can be accessed at: [Kaggle Dataset Link](https://www.kaggle.com/fedesoriano/stroke-prediction-dataset) and emerged as the most suitable choice due to its alignment with the primary focus of stroke prediction and prevention. This dataset encompasses a wide array of attributes, including demographic information, medical history, and lifestyle factors.")
    st.write("This dataset comprises 5110 records and 12 columns featuring both numerical and categorical data. It includes critical information such as unique identifiers, gender, age, medical conditions (hypertension and heart disease), marital status, occupation, residence type, glucose levels, BMI, smoking status, and stroke occurrences. Its primary objective is to unveil relationships between these factors and the likelihood of a stroke.")

    st.markdown("""<hr style="height:3px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
    st.write("If you'd like to view the unprocessed data, click the 'Show Raw Data' button. For those interested in numbers, explore detailed feature breakdowns and statistical analysis in the section below.")
    
    checks = st.columns(2)
    # Display the dataset
    with checks[0]:
        with st.expander("Show Raw Data"):
            # if st.checkbox('Show Raw Data'):
            st.write(pd.DataFrame(df, columns=df.columns))
            st.write('Stroke Prediction Dataset Information:')
            st.write(f'Total Number of Samples: {df.shape[0]}')
            st.write(f'Number of Features: {df.shape[1]}')

    with checks[1]:
        with st.expander("Show Statistics about Data"):
            st.write(df.describe())
            st.write('Stroke Prediction Dataset Information:')
            st.write(f'Total Number of Samples: {df.shape[0]}')
            st.write(f'Number of Features: {df.shape[1]}')

    st.write('To explore the data further we will take a look into interactive plots and visualizations in the next tabs')
        


with tab2 :
    st.sidebar.title('Welcome to the data exploration section')
    st.header("What factors are causing a Stroke ?")

    replace_missing_with_median(df)

    # Sidebar inputs
    st.sidebar.subheader('Use filters to uncover insights')
    selected_work_type = st.sidebar.selectbox('Work Type', df['work_type'].unique())
    selected_smoking_status = st.sidebar.selectbox('Smoking Status', df['smoking_status'].unique())
    selected_age_range = st.sidebar.slider('Age Range', int(df['age'].min()), int(df['age'].max()), (20, 80))
    selected_gender = st.sidebar.selectbox('Gender', df['gender'].unique())

    # Apply filters and store filtered data
    filtered_data = filter_data(df, selected_work_type, selected_smoking_status, selected_age_range, selected_gender)

    # Display filtered data
    st.sidebar.subheader('Filtered Data')
    st.sidebar.dataframe(filtered_data)

    if st.checkbox('Examining stroke trends by lifestyle category'):
        # Partition and description for bar graph
        st.markdown("---")
        st.subheader('Examining stroke trends by lifestyle category.')
        st.markdown("You can pick different categories like gender, marital status, type of work, where you live, and smoking habits. It's a bit like a colorful bar chart you might see in a magazine. This chart helps you see how many people in each category had a stroke. By looking at this chart, you can figure out if these things, like being married or where you live, can make a stroke more likely to happen. It's like a detective tool to study strokes and learn what might cause them.")

        # Create bar plot
        bar_x = st.selectbox('Select a category', df.select_dtypes(include=['object']).columns)
        bar_plot = create_bar_plot(df, bar_x)
        st.plotly_chart(bar_plot)

        # Description for bar graph
        if bar_x == 'smoking_status':
            st.markdown("**Smoking Status:** By looking at the bar graph for the 'smoking status' category, we can learn about the relationship between smoking and strokes. The graph shows different groups, like people who smoke, used to smoke, or never smoked. The height of each bar represents how many people in each group had a stroke. So, if you see a really tall bar for the 'smokers' group, it means a lot of smokers had strokes. If the 'never smoked' bar is short, it suggests that fewer non-smokers had strokes.")
        elif bar_x == 'work_type':
            st.markdown("**Work Type:** From the bar graph of the 'Work Type' category, we can see how different types of jobs are related to strokes. The graph uses different colors to show the number of people in each job category who had a stroke. By looking at this graph, we can understand if certain jobs have a higher or lower risk of strokes. ")
        elif bar_x == 'gender':
            st.markdown("**Gender:**  In this graph, you'll see two bars, one for 'Male' and one for 'Female'. The height of these bars shows how many men and women in the dataset had a stroke. If one bar is taller than the other, it means more people of that gender had a stroke. This helps us understand if gender has any connection to strokes. For example, if the 'Female' bar is taller, it might mean that women in this dataset had more strokes. ")
        elif bar_x == 'Residence_type':
            st.markdown("**Residence Type:**  From the bar graph of the 'Residence Type' category, you can see two colorful bars representing two different types of residence: 'Urban' and 'Rural.' Each bar shows the number of people who had a stroke in these two types of areas. This helps us understand whether living in an urban or rural place might be connected to having a stroke. It's like looking at the numbers to see if the place where someone lives is related to their risk of having a stroke. This information can be valuable for understanding and preventing strokes.")
        elif bar_x == 'ever_married':
            st.markdown("**Ever Married:**   The graph will show two bars, one for people who have been married and another for those who haven't. You'll see how many people in each group had a stroke. This helps us understand if being married or not being married might affect the chances of having a stroke. It's like looking at data to find clues about how different life experiences might impact our health.")

        st.markdown("Observations:")
        st.markdown("1. When we look at features like gender and residence type, they don't seem to make a big difference in predicting whether a person will have a stroke or not. The chances of having a stroke for different groups within these features are quite similar to the overall dataset.")
        st.markdown("3. As for features like marriage status, work type, and smoking habits, the chances of having a stroke for specific groups are noticeably different from the overall dataset. This indicates that these features are also important in determining whether a person is at risk of having a stroke.")
    
    if st.checkbox('Examining stroke trends with human charcateristics'): 
        # Partition and description for violin graph
        st.markdown("---")
        st.subheader('Examining stroke trends with human charcateristics')
        st.markdown("You can pick different factors like age, average glucose level, and BMI. The violin graph here shows a picture of how these numbers are spread out among people who had a stroke and people who didn't. It helps us compare and see the differences in these factors between the two groups. It's like a visual tool to understand how these numbers affect the chances of having a stroke.")

        # Create violin plot
        violin_y = st.selectbox('Select a category', df.select_dtypes(include=['float64']).columns)
        violin_plot = create_violin_plot(df, violin_y)
        st.plotly_chart(violin_plot)

        # Description for violin graph
        if violin_y == 'bmi':
            st.markdown("**BMI (Body Mass Index):** By examining this graph, we can draw some valuable conclusions. If you notice that there's a significant difference in BMI between people who had a stroke and those who didn't, it could suggest that BMI plays a role in stroke risk. In simpler terms, it helps us see if being underweight, overweight, or having a healthy BMI might affect the likelihood of experiencing a stroke. So, if you see a big difference in the heights of the bars in the graph, it might indicate that BMI is an important factor when it comes to stroke prediction.")
        elif violin_y == 'avg_glucose_level':
            st.markdown("**Average Glucose Level:** The graph displays how average glucose levels are distributed among individuals in the dataset who either had a stroke or did not. You can see two violins side by side, one for those who had a stroke and one for those who didn't. By looking at the shape and position of these violins, you can make inferences. For example, if the 'stroke' violin is noticeably wider at higher glucose levels, it might suggest that elevated glucose levels could be associated with a higher risk of stroke.")
        elif violin_y == 'age':
            st.markdown("**Age:** The width of the plot shows us how ages are distributed among people who either had a stroke or did not. If the plot is wider at a certain age range, it means there are more people in that group. For example, if the plot is wider in the middle, it means more people in that age group had a stroke. By looking at the plot, we can see if there's a particular age group where strokes are more common. This helps us understand how age is related to the risk of having a stroke.")

        st.markdown("Observations:")
        st.markdown("1. For younger patients, especially those aged 0-30, the chances of having a stroke are very low (less than 0.01), but there's a significant increase in the likelihood for patients aged 60 and older.")
        st.markdown("2. Patients with BMI in the range of 20-50 have stroke probabilities similar to the overall dataset. However, there's a noticeable drop in stroke probability for patients with BMIs outside this range. Please note that some groups may have small sample sizes.")
        st.markdown("3. Patients with lower average glucose levels (below 170) have stroke probabilities similar to the overall dataset. In contrast, patients with higher average glucose levels have a significantly higher stroke probability.")

    if st.checkbox('Histogram'):
        st.header('Histogram')
        st.write(" Imagine the bars as groups, each with a different color. These bars show how often something happened, like a crisis, and how it's related to something else, like a financial factor. The taller the bars, the more it happened, and you can point your mouse at them to see the exact numbers. When the bars overlap, like they're close together, it means these things are connected. If one group's bars are mostly on one side and another group's bars are on the other side, it means they're different in some way. It's like they're sharing a secret!")
        col1,col2=st.columns(2,gap='small')
        st.subheader('Select a feature for the histogram:')
        selected_feature = st.selectbox('Select a feature', df.select_dtypes(include=['float64']).columns)
        gender_format = 'gender'
        bin_count = st.slider('Number of Bins', min_value=1, max_value=100, value=20)

        st.subheader(f'Histogram of {selected_feature}')

        # Create an interactive histogram using Plotly Express
        fig = px.histogram(df, x=selected_feature, color=gender_format, nbins=bin_count)
        fig.update_xaxes(title_text=selected_feature)
        fig.update_yaxes(title_text='Count')
        fig.update_traces(marker=dict(line=dict(width=2)))
        fig.update_layout(height=700, width=900)
        # Display the interactive plot
        st.plotly_chart(fig)

        st.markdown("Observations:")
        st.markdown("The patterns for average glucose levels and BMI look a bit like a bell curve, with a slight tilt to the right. This means there are a few higher values on the right side. Similar behavior is seen for age, where there's also a bit of spread. There isn't much distinction between genders. However, we have very few samples from other gender groups, so we won't consider them for now.")

    if st.checkbox('Studying relationships in stroke data'):  
        st.subheader("Studying relationships in stroke data.")
        st.markdown("Let's explore a new concept that can help us understand the factors related to strokes. We're going to look at something called a 'Bivariate scatterplot by diagnosis (Stroke).' This is like a special picture that shows two things at once and helps us see if there's a connection between those two things and whether someone had a stroke.")
        st.markdown("In simpler terms, it's like using a visual chart to see how two different factors might be linked to strokes. For instance, we can use this chart to investigate whether age and having high blood pressure are connected to having a stroke. The chart uses dots to show this relationship, making it much easier for us to see and understand the connections. So, let's dive into this visual tool and uncover valuable insights about strokes and the factors involved.")
        
        l1, m1, r1 = st.columns((2,5,1))

        col3, col4, col5 = st.columns(3,gap='large')

        numerical = df.select_dtypes(include=['float64']).columns;
        categorical = df.select_dtypes(include=['object']).columns;
        with col3:
            alt_x = st.selectbox("Select a feature for (X)?", numerical)
        with col4:
            alt_y = st.selectbox("Select a feature for (Y) ?", numerical)
        with col5:
            cat_hue = st.selectbox("Choose target", categorical)

        if alt_x and alt_y and cat_hue:
            fig3 = px.scatter(df, alt_x, alt_y, color=cat_hue, trendline="ols")
            fig3.update_layout({
            'plot_bgcolor': 'rgba(0, 0, 0, 0)',
            'paper_bgcolor': 'rgba(0, 0, 0, 0)',
            },font=dict(
                size=18
                )   
            )

        st.write(fig3)

        st.markdown('When you study these scatter plots, you can make some interesting discoveries:')
        st.markdown("1. Lines: When you notice a line going up or down, it's like seeing a connection between the two factors you chose. If it's going up, it means when one thing increases, the other often increases too. If it's heading down, when one thing goes up, the other tends to go down.")
        st.markdown("2. Dots: Think of each dot as representing one person's information. If most of these dots are close to the line, it tells us that these two factors are closely linked. On the other hand, if the dots are spread out all over the place, it suggests that they aren't strongly connected.")
        st.markdown("So, these plots act like visual puzzles. If you spot a clear pattern in the dots, it helps you understand how these factors affect each other. It's like uncovering clues in a colorful puzzle!")
    
        st.markdown("Observations:")
        st.markdown("The scatter plots suggest that as people get older, their average glucose levels and body mass index (BMI) tend to rise, and so does their risk of having a stroke. Additionally, it appears that age and average glucose levels have a more significant impact on the risk of a stroke.")

    if st.checkbox('Correlation'): 

        l3, m3, r3 = st.columns((4, 5, 1))
        st.subheader("Correlation")

        st.markdown("Correlation means looking at how two things are connected or related to each other. For example, imagine we want to understand if age and the risk of having a stroke are connected. If we find that, as people get older, the chances of having a stroke also increase, we say there's a positive correlation between age and stroke risk. On the other hand, if we see that as the number of fruits people eat increases, the risk of a stroke decreases, that would be a negative correlation.")

        st.markdown("Correlation helps us figure out if one thing might be influencing another, like age influencing stroke risk, or if there's no connection between them at all.")    

        with st.form("key2"):
            corr_range = st.slider("Select correlation magnitude range", value=[-1.0, 1.0], step=0.05)

            correlation_data = create_correlation_matrix(df, corr_range)

            st.write("Correlation Matrix:")
            st.dataframe(correlation_data, width=800, height=150)

            button2 = st.form_submit_button("Apply range")

        corr_mat = st.checkbox('Show/hide correlation matrix')

        if corr_mat:
            st.subheader("Correlation Matrix Heatmap")
            st.markdown("The correlation matrix heatmap provides an overview of the relationships between numerical features. Strong correlations are shown in warmer (reddish) or cooler (bluish) colors.")

            fig_corr = px.imshow(correlation_data,
                                color_continuous_scale="RdBu_r",
                                title="Correlation Matrix Heatmap")
            fig_corr.update_layout(width=800, height=600)
            st.plotly_chart(fig_corr)

            st.markdown("Observations:")
            st.markdown("The colorful heatmap reveals that there are only faint links between average glucose levels and age, as well as average glucose levels and BMI (Body Mass Index). However, there's a somewhat stronger connection between BMI and age.")    


with tab3 :
    #visualization with HiPlot
    def save_hiplot_to_html(exp):
        output_file = "hiplot_plot_1.html"
        exp.to_html(output_file)
        return output_file
    st.write("Visualization with HiPlot")
    selected_columns = st.multiselect("Select columns to visualize", df.columns)
    selected_data = df[selected_columns]
    if not selected_data.empty:
        experiment = hip.Experiment.from_dataframe(selected_data)
        hiplot_html_file = save_hiplot_to_html(experiment)
        st.components.v1.html(open(hiplot_html_file, 'r').read(), height=1500, scrolling=True)
    else:
        st.write("No data selected. Please choose at least one column to visualize.")
