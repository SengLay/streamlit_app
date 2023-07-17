import streamlit as st
import pandas as pd
from pathlib import Path
import base64
import io
import os

def table_of_content():
    # Table of Content
    st.sidebar.title("Table of Contents")
    st.sidebar.write("[1. Introduction](#introduction)")
    st.sidebar.write("[2. Group Members](#group-members)")
    st.sidebar.write("[3. Objectives of Job Prediction Project](#objectives)")
    st.sidebar.write("[4. Data Exploration](#data-exploration)")
    st.sidebar.write("[4.1. Importing Python Language Libraries](#import-dataset)")
    st.sidebar.write("[4.2. The Original Dataset AllPhnomList.csv File](#origin-dataset)")
    st.sidebar.write("[4.3. Data Description](#data-description)")
    st.sidebar.write("[4.3.1. Dataset Info](#info-data)")
    st.sidebar.write("[4.3.2. Categorical Features](#cat-features)")
    st.sidebar.write("[4.3.3. Numerical Features](#num-features)")
    st.sidebar.write("[4.4. Data Cleaning](#data-cleaning)")
    st.sidebar.write("[4.4.1. Missing Values](#missing-vals)")
    st.sidebar.write("[4.4.2. Duplicated Values](#dup-vals)")
    st.sidebar.write("[4.4.3. Outlier](#outlier)")
    st.sidebar.write("[5. Exploratory Data Analysis (EDA)](#eda)")
    st.sidebar.write("[5.1. Job Type](#job-type)")
    st.sidebar.write("[5.2. Position Level](#pos-lvl)")
    st.sidebar.write("[5.3. Location](#loc)")
    st.sidebar.write("[5.4. Working Experience](#work-exp)")
    st.sidebar.write("[5.5. Qualification](#qualification)")
    st.sidebar.write("[5.6. Age](#age)")
    st.sidebar.write("[5.7. Salary](#salary)")
    st.sidebar.markdown("[Project Report](#report)")
    st.sidebar.markdown("[Project Guideline](#guideline)")
table_of_content()

def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

def load_data(file_path):
    # Use os.path.join to create the file path
    data_path = os.fspath(file_path)

    # Check if the file exists
    if os.path.exists(data_path):
        return data_path
    else:
        st.error("File not found: {}".format(data_path))

outlier_path = load_data("Photo/outlier.png")
def outlier():
    outlier_img = "<center><figure><img src='data:image/png;base64,{}' class='img-fluid' width=700><figcaption>Outlier Graph</figcaption></figure></center>".format(
        img_to_bytes(outlier_path)
    )
    container = st.container()
    with container:
        st.markdown(outlier_img, unsafe_allow_html=True)

# Job Type
jobtype_1_path = load_data("Photo/jobtype-1.png")
def jobtype_1():
    jobtype1_img = "<center><figure><img src='data:image/png;base64,{}' class='img-fluid' width=650><figcaption>Job Type Graph</figcaption></figure></center>".format(
        img_to_bytes(jobtype_1_path)
    )
    container = st.container()
    with container:
        st.markdown(jobtype1_img, unsafe_allow_html=True)

jobtype_2_path = load_data("Photo/jobtype-2.png")
def jobtype_2():
    jobtype2_img = "<center><figure><img src='data:image/png;base64,{}' class='img-fluid' width=600><figcaption>Job Type Graph</figcaption></figure></center>".format(
        img_to_bytes(jobtype_2_path)
    )
    container = st.container()
    with container:
        st.markdown(jobtype2_img, unsafe_allow_html=True)

jobtype_3_path = load_data("Photo/jobtype-3.png")
def jobtype_3():
    jobtype3_img = "<center><figure><img src='data:image/png;base64,{}' class='img-fluid' width=600><figcaption>Job Type Graph</figcaption></figure></center>".format(
        img_to_bytes(jobtype_3_path)
    )
    container = st.container()
    with container:
        st.markdown(jobtype3_img, unsafe_allow_html=True)

# Position Level
pos_1_path = load_data("Photo/poslevel-1.png")
def poslevel_1():
    poslevel1_img = "<center><figure><img src='data:image/png;base64,{}' class='img-fluid' width=600><figcaption>Distribution of Position Level Graph</figcaption></figure></center>".format(
        img_to_bytes(pos_1_path)
    )
    container = st.container()
    with container:
        st.markdown(poslevel1_img, unsafe_allow_html=True)

pos_2_path = load_data("Photo/poslevel_vs_sal.png")
def poslevel_vs_sal():
    poslevel_sal_img = "<center><figure><img src='data:image/png;base64,{}' class='img-fluid' width=600><figcaption>Distribution of Position Level Graph with Salary</figcaption></figure></center>".format(
        img_to_bytes(pos_2_path)
    )
    container = st.container()
    with container:
        st.markdown(poslevel_sal_img, unsafe_allow_html=True)

# Location
loc_1_path = load_data("Photo/loc_freq.png")
def loc_freq_count():
    loc_freq = "<center><figure><img src='data:image/png;base64,{}' class='img-fluid' width=600><figcaption>Location Frequency Count</figcaption></figure></center>".format(
        img_to_bytes(loc_1_path)
    )
    container = st.container()
    with container:
        st.markdown(loc_freq, unsafe_allow_html=True)

loc_2_path = load_data("Photo/abbre_loc.png")
def abbrev_loc():
    loc_abbre = "<center><figure><img src='data:image/png;base64,{}' class='img-fluid' width=650><figcaption>Distributions of Location after Abbreviation</figcaption></figure></center>".format(
        img_to_bytes(loc_2_path)
    )
    container = st.container()
    with container:
        st.markdown(loc_abbre, unsafe_allow_html=True)

loc_3_path = load_data("Photo/sal_vs_loc.png")
def sal_vs_loc():
    sal_loc_img = "<center><figure><img src='data:image/png;base64,{}' class='img-fluid' width=650><figcaption>Relationship between Minimum Salary and Location</figcaption></figure></center>".format(
        img_to_bytes(loc_3_path)
    )
    container = st.container()
    with container:
        st.markdown(sal_loc_img, unsafe_allow_html=True)

# Work Experience
exp_1_path = load_data("Photo/work_exp_dis.png")
def work_exp():
    workexp_img = "<center><figure><img src='data:image/png;base64,{}' class='img-fluid' width=650><figcaption>Distribution of Working Experience</figcaption></figure></center>".format(
        img_to_bytes(exp_1_path)
    )
    container = st.container()
    with container:
        st.markdown(workexp_img, unsafe_allow_html=True)

# Qualification
qual_1_path = load_data("Photo/qua_dis.png")
def qualification_dis():
    qua_dis_img = "<center><figure><img src='data:image/png;base64,{}' class='img-fluid' width=650><figcaption>Distribution of Qualification</figcaption></figure></center>".format(
        img_to_bytes(qual_1_path)
    )
    container = st.container()
    with container:
        st.markdown(qua_dis_img, unsafe_allow_html=True)

qual_2_path = load_data("Photo/qua_vs_sal.png")
def qualification_vs_sal():
    qua_vs_sal_img = "<center><figure><img src='data:image/png;base64,{}' class='img-fluid' width=650><figcaption>Relationship Between Qualification and Average Minimum Salary</figcaption></figure></center>".format(
        img_to_bytes(qual_2_path)
    )
    container = st.container()
    with container:
        st.markdown(qua_vs_sal_img, unsafe_allow_html=True)

qual_3_path = load_data("Photo/qua_abbre.png")
def qualification_dis_after_abbre():
    qua_vs_sal_abbre_img = "<center><figure><img src='data:image/png;base64,{}' class='img-fluid' width=650><figcaption>Distribution of Qualifications</figcaption></figure></center>".format(
        img_to_bytes(qual_3_path)
    )
    container = st.container()
    with container:
        st.markdown(qua_vs_sal_abbre_img, unsafe_allow_html=True)

age_1_path = load_data("Photo/age.png")
def age_dis():
    age_dis_img = "<center><figure><img src='data:image/png;base64,{}' class='img-fluid' width=650><figcaption>Distribution of Minimum Ages</figcaption></figure></center>".format(
        img_to_bytes(age_1_path)
    )
    container = st.container()
    with container:
        st.markdown(age_dis_img, unsafe_allow_html=True)

age_2_path = load_data("Photo/age_sal.png")
def age_vs_sal():
    age_vs_sal_img = "<center><figure><img src='data:image/png;base64,{}' class='img-fluid' width=650><figcaption>Relationship between Minimum Age and Average Minimum Salary</figcaption></figure></center>".format(
        img_to_bytes(age_2_path)
    )
    container = st.container()
    with container:
        st.markdown(age_vs_sal_img, unsafe_allow_html=True)

# Salary
sal_1_path = load_data("Photo/min_sal_dis.png")
def sal_dis():
    sal_dis_img = "<center><figure><img src='data:image/png;base64,{}' class='img-fluid' width=650><figcaption>Distribution of Minimum Salaries</figcaption></figure></center>".format(
        img_to_bytes(sal_1_path)
    )
    container = st.container()
    with container:
        st.markdown(sal_dis_img, unsafe_allow_html=True)

sal_2_path = load_data("Photo/pairwise_sal.png")
def pairwise_sal():
    pair_sal = "<center><figure><img src='data:image/png;base64,{}' class='img-fluid' width=650><figcaption>Pairwise Scatter Plot</figcaption></figure></center>".format(
        img_to_bytes(sal_2_path)
    )
    container = st.container()
    with container:
        st.markdown(pair_sal, unsafe_allow_html=True)

sal_3_path = load_data("Photo/boxplot_cat_vars.png")
def boxplot_cat_sal():
    box_cat_sal = "<center><figure><img src='data:image/png;base64,{}' class='img-fluid' width=650><figcaption>Boxplot For Categorical Variables</figcaption></figure></center>".format(
        img_to_bytes(sal_3_path)
    )
    container = st.container()
    with container:
        st.markdown(box_cat_sal, unsafe_allow_html=True)

sal_4_path = load_data("Photo/pair_age_sal.png")
def pairwise_age_sal():
    pair_age_sal = "<center><figure><img src='data:image/png;base64,{}' class='img-fluid' width=650><figcaption>Relationship between Minimum Age and Minimum Salary</figcaption></figure></center>".format(
        img_to_bytes(sal_4_path)
    )
    container = st.container()
    with container:
        st.markdown(pair_age_sal, unsafe_allow_html=True)

sal_5_path = load_data("Photo/subplots_sal.png")
def subplots_sal():
    sub_sal = "<center><figure><img src='data:image/png;base64,{}' class='img-fluid' width=650><figcaption>Subplots with Minimum Salary</figcaption></figure></center>".format(
        img_to_bytes(sal_5_path)
    )
    container = st.container()
    with container:
        st.markdown(sub_sal, unsafe_allow_html=True)

def image():
    itc_path = load_data("Photo/itc.png")
    header_html_1 = "<center><figure><img src='data:image/png;base64,{}' class='img-fluid' width=250><figcaption>Institute of technology of Cambodia</figcaption></figure></center>".format(
        img_to_bytes(itc_path)
    )
    ams_path = load_data("Photo/dep.jpg")
    header_html_2 = "<center><figure><img src='data:image/png;base64,{}' class='img-fluid' width=246><figcaption>Department of Applied Mathematics and Statistics</figcaption></figure></center>".format(
        img_to_bytes(ams_path)
    )
    ministry_path = load_data("Photo/moey.png")
    header_html_3 = "<center><figure><img src='data:image/png;base64,{}' class='img-fluid' width=183><figcaption>Ministry of Education, Youth and Sport (Cambodia)</figcaption></figure></center>".format(
        img_to_bytes(ministry_path)
    )
    container = st.container()
    with container:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(header_html_2, unsafe_allow_html=True)
        with col2:
            st.markdown(header_html_1, unsafe_allow_html=True)
        with col3:
            st.markdown(header_html_3, unsafe_allow_html=True)
image()

def welcome_project_description():
    # Welcome Statement & Project Description
    st.write(
        '''
        # <center style="color:black;">Welcome to Job Prediction Project!</center>
        ''', unsafe_allow_html=True
    )

    # Project Description
    st.markdown("""<div id="introduction"></div>""", unsafe_allow_html=True)
    st.header("1. Introduction")
    st.markdown(
        '''
        <div style="border: 1px solid black; padding: 10px; width: 100%; box-shadow: 5px 5px 5px #888888; background-color: AliceBlue; color: SlateGrey; font-size: 16px; text-align: center;">

        Job Prediction created in order of to analyze some features that we got from the previous Web Scraping Project.
        Our project focuses on developing an advanced job prediction model that accurately forecasts salaries in the ever-changing 
        job market. As a group from Institute of Technology of Cambodia, our team comprises talented individuals from various 
        educational backgrounds. <b><i>SENG Lay, VANNAK Vireakyuth, YA Manon, VANN Visal, TAING Kimmeng,</i></b> and <b><i>VINLAY Anusar</i></b> members of our group, 
        bring their expertise and insights to the project. With a deep understanding of the intricacies of the job market and the latest trends, we aim to revolutionize the way individuals and 
        organizations make decisions regarding career paths and salary negotiations. By leveraging cutting-edge algorithms and 
        comprehensive data analysis, our project aims to provide accurate and reliable salary predictions that empower job seekers, 
        employers, and educational institutions. Together, we are committed to creating a game-changing solution that transforms the 
        landscape of job prediction and salary forecasting.

        <b>GitHub Source Code for Job Prediction:</b> <i><b>[Click Here!](https://github.com/SengLay/Job-Analysis)</b></i></div>
        ''', unsafe_allow_html=True
    )
welcome_project_description()
lay_path = load_data("Photo/lay copy.jpg")
lay = "<center><figure><img src='data:image/png;base64,{}' class='img-fluid' width=300><figcaption></figcaption></figure></center>".format(
    img_to_bytes(lay_path)
)
manon_path = load_data("Photo/manon.jpg")
manon = "<center><figure><img src='data:image/png;base64,{}' class='img-fluid' width=300><figcaption></figcaption></figure></center>".format(
    img_to_bytes(manon_path)
)
visal_path = load_data("Photo/visal.jpg")
visal = "<center><figure><img src='data:image/png;base64,{}' class='img-fluid' width=300><figcaption></figcaption></figure></center>".format(
    img_to_bytes(visal_path)
)
anusar = "<center><figure><img src='data:image/png;base64,{}' class='img-fluid' width=300><figcaption></figcaption></figure></center>".format(
    img_to_bytes("Photo/anusar.jpg")
)
yuth = "<center><figure><img src='data:image/png;base64,{}' class='img-fluid' width=300><figcaption></figcaption></figure></center>".format(
    img_to_bytes("Photo/yuth.jpg")
)
kimmeng = "<center><figure><img src='data:image/png;base64,{}' class='img-fluid' width=300><figcaption></figcaption></figure></center>".format(
    img_to_bytes("Photo/kimmeng.jpg")
)

def members():
    # Group Members Image and Background
    st.markdown("""<div id="group-members"></div>""", unsafe_allow_html=True)
    st.header("2. Group Members")
    with st.expander("CLICK HERE TO SEE OUR GROUP MEMBERS", expanded=True):

        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["SENG Lay", "VANNAK Vireakyuth", "YA Manon", "VANN Visal", "TAING Kimmeng", "VINLAY Anusar"])
        with tab1:
            st.markdown("<div style='font-size: 40px';><center><b>SENG Lay</b></center></div>",
                        unsafe_allow_html=True)
            st.markdown(lay, unsafe_allow_html=True)
            st.markdown("<div style='font-size: 25px';><center><b>Leader</b></center></div>", unsafe_allow_html=True)

        with tab2:
            st.markdown("<div style='font-size: 40px';><center><b>VANNAK Vireakyuth</b></center></div>",
                        unsafe_allow_html=True)
            st.markdown(yuth, unsafe_allow_html=True)
            st.markdown("<div style='font-size: 25px';><center><b>Member</b></center></div>",
                        unsafe_allow_html=True)

        with tab3:
            st.markdown("<div style='font-size: 40px';><center><b>YA Manon</b></center></div>",
                        unsafe_allow_html=True)
            st.markdown(manon, unsafe_allow_html=True)
            st.markdown("<div style='font-size: 25px';><center><b>Member</b></center></div>",
                        unsafe_allow_html=True)

        with tab4:
            st.markdown("<div style='font-size: 40px';><center><b>VANN Visal</b></center></div>",
                        unsafe_allow_html=True)
            st.markdown(visal, unsafe_allow_html=True)
            st.markdown("<div style='font-size: 25px';><center><b>Member</b></center></div>",
                        unsafe_allow_html=True)

        with tab5:
            st.markdown("<div style='font-size: 40px';><center><b>TAING Kimmeng</b></center></div>",
                        unsafe_allow_html=True)
            st.markdown(kimmeng, unsafe_allow_html=True)
            st.markdown("<div style='font-size: 25px';><center><b>Member</b></center></div>",
                        unsafe_allow_html=True)

        with tab6:
            st.markdown("<div style='font-size: 40px';><center><b>VINLAY Anusar</b></center></div>",
                        unsafe_allow_html=True)
            st.markdown(anusar, unsafe_allow_html=True)
            st.markdown("<div style='font-size: 25px';><center><b>Member</b></center></div>",
                        unsafe_allow_html=True)
members()

def obj():
    # Objective of project
    st.markdown("""<div id="objectives"></div>""", unsafe_allow_html=True)
    st.header("3. Objectives of Job Prediction Project")
    st.markdown(
        '''
        The development and deployment of a machine learning model for salary prediction offer several benefits:

        * Accurate salary prediction based on features such as interpersonal skills, work experience, education, weekly working days, employment type (full-time or part-time), age, and location.
        * Facilitation of fair compensation practices by considering multiple factors and minimizing salary discrepancies.
        * Streamlining of the hiring process by estimating appropriate salary ranges for job applicants.
        * Improved efficiency in recruitment efforts by swiftly identifying candidates whose salary expectations align with the organization's offerings.
        * Empowerment of organizations in workforce planning and resource allocation decisions.
        * Enhanced decision-making related to talent acquisition, ensuring fair salary distribution, and optimizing compensation structures.
        * Provision of valuable insights through analysis of the relationships between salary and various factors such as education, experience, and location.
        * Identification of patterns and trends for informed decision-making in talent acquisition and retention.
        * Gain a competitive edge in the job market by adapting compensation strategies to attract and retain top talent.
        '''
    )
obj()

def data_exploration_4():
    ### Data Exploration ###
    st.write("## `LET'S DIVE INTO OUR PROJECT!`")
    st.markdown("""<div id="data-exploration"></div>""", unsafe_allow_html=True)
    st.header("4. Data Exploration: Exploring the Dataset")

    # Import libraries
    st.markdown("""<div id="import-dataset"></div>""", unsafe_allow_html=True)
    st.subheader('4.1. Importing Python Language Libraries')
    with st.expander("CLICK HERE TO SHOW DATA"):
        st.write(
            """
            Bonjour, First of all we need to import important libraries needed for our project first! <3
            """
        )
        code = '''
            import pandas as pd
            import numpy as np
            import seaborn as sns
            import matplotlib.pyplot as plt
            %matplotlib inline
            import statsmodels.api as sm 
            from sklearn import metrics
            from sklearn.preprocessing import MinMaxScaler
            from sklearn.model_selection import train_test_split
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
            from statsmodels.stats.outliers_influence import variance_inflation_factor
            from sklearn.feature_selection import RFE
            from sklearn.preprocessing import LabelEncoder
            import warnings
            warnings.filterwarnings("ignore")
            Display all the column of the dataframes
            pd.pandas.set_option('display.max_columns', None)
            '''
        st.code(code, language='python')

    # Load dataset
    st.markdown("""<div id="origin-dataset"></div>""", unsafe_allow_html=True)
    st.subheader('4.2. The Original Dataset `AllPhnomList.csv` file')
    with st.expander("CLICK HERE TO SHOW DATA"):
        st.code(
            '''
            df = pd.read_csv('CSV/data2.csv')
            df
            ''', language='python'
        )
        st.write(
            """
            This is the original dataset of our project.
            """
        )
        df = pd.read_csv('CSV/data2.csv')
        st.dataframe(df)

    # Describe the whole dataset
    st.markdown("""<div id="data-description"></div>""", unsafe_allow_html=True)
    st.subheader('4.3. Data Description')
    with st.expander("CLICK HERE TO SHOW DATA"):
        st.code(
            '''
            df.describe()
            ''', language='python'
        )
        st.write(
            """
            Take a look at the desciption of our dataset here.
            """
        )
        list_stats = df.describe()
        st.dataframe(list_stats)

    # Dataset Info
    st.markdown("""<div id="info-data"></div>""", unsafe_allow_html=True)
    st.subheader('4.3.1. Dataset Info')
    with st.expander("CLICK HERE TO SHOW DATA"):
        st.code(
            '''
            df.info()
            ''', language='python'
        )
        st.write(
            """
            Take a look at the information of our dataset here.
            """
        )
        buffer = io.StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)

    # Categorical Features
    st.markdown("""<div id="cat-features"></div>""", unsafe_allow_html=True)
    st.subheader('4.3.2. Categorical Features')
    with st.expander("CLICK HERE TO SHOW DATA"):
        st.code(
            '''
            cat_features = [col for col in df.columns if df[col].dtypes == 'O']
            cat_features, len(cat_features)
            ''', language='python'
        )
        st.write(
            """
            Take a look at the Categorical Features of our dataset here.
            """
        )
        cat_features = [col for col in df.columns if df[col].dtypes == 'O']
        cat_features, len(cat_features)
        st.text(f"Categorical Features: {cat_features}")
        st.text(f"Length of Categorical Features: {len(cat_features)}")

    # Numerical Features
    st.markdown("""<div id="num-features"></div>""", unsafe_allow_html=True)
    st.subheader('4.3.3. Numerical Features')
    with st.expander("CLICK HERE TO SHOW DATA"):
        st.code(
            '''
            num_features = [col for col in df.columns if col not in cat_features]
            num_features, len(num_features)
            ''', language='python'
        )
        st.write(
            """
            Take a look at the Numerical Features of our dataset here.
            """
        )
        num_features = [col for col in df.columns if col not in cat_features]
        num_features, len(num_features)
        st.text(f"Numerical Features: {num_features}")
        st.text(f"Length of Numerical Features: {len(num_features)}")
data_exploration_4()

def data_cleaning():
    st.markdown("""<div id="data-cleaning"></div>""", unsafe_allow_html=True)
    st.subheader('4.4. Data Cleaning')

    ### 4.4.1
    st.markdown("""<div id="missing-vals"></div>""", unsafe_allow_html=True)
    st.subheader('4.4.1. Missing Values')
    with st.expander("CLICK HERE TO SHOW DATA"):
        st.code(
            '''
            # Drop unnescessaries features and cleaning 
            df.drop(columns=['Salary_max', 'max_age'], inplace=True, axis =1)

            # Check Missing Values
            df.isnull().sum()
            ''', language='python'
        )
        st.write(
            """
            Take a look at the result after checking.
            """
        )
        df = pd.read_csv('CSV/data2.csv')
        s = df.isnull().sum()
        st.text(s)

    ### 4.4.2
    st.markdown("""<div id="dup-vals"></div>""", unsafe_allow_html=True)
    st.subheader('4.4.2. Duplicated Values')
    with st.expander("CLICK HERE TO SHOW DATA"):
        st.code(
            '''
            # Sum up all duplicated values
            df.duplicated().sum()
            ''', language='python'
        )

        duplicated_sum = df.duplicated().sum()
        st.text(f'Duplicated values: {duplicated_sum}')

        st.code(
            '''
            # Drop all duplicated values
            df.drop_duplicates(inplace= True)
            ''', language='python'
        )

    ### 4.4.3
    st.markdown("""<div id="outlier"></div>""", unsafe_allow_html=True)
    st.subheader('4.4.3. Outlier')
    with st.expander("CLICK HERE TO SHOW DATA"):
        st.code(
            '''
            # Plot the graph using seaborn to check the outlier
            plt.figure(figsize=(14, 9))
            sns.boxplot(data = df)
            plt.savefig('Photo/outlier.png')
            plt.show()
            ''', language='python'
        )
        st.write(
            """
        Outlier Graph using seaborn
            """
        )
        outlier()
data_cleaning()

def EDA_JobType():
    st.markdown("""<div id="eda"></div>""", unsafe_allow_html=True)
    st.header("5. Exploratory Data Analysis (EDA)")
    st.write('''
    In this EDA section, we're going to check details on each variables in our dataset.
    ''')

    st.markdown("""<div id="job-type"></div>""", unsafe_allow_html=True)
    st.subheader("5.1. Job Type")
    with st.expander("CLICK HERE TO SHOW DATA"):
        # Values Count
        st.write('##### Minimum Salary Values Counts')
        st.code(
            '''
            df.Salary_min.value_counts()
            ''', language='python'
        )
        st.write(
            """
            Take a look at the result after checking.
            """
        )
        df = pd.read_csv('CSV/data2.csv')
        s = df.Salary_min.value_counts()
        st.text(s)

        # Distribution of Job Type
        st.write('##### Distribution of each Job Type')
        st.code(
            '''
            # Get unique categories in JobType
            job_types = df['JobType'].unique()

            # Define the number of rows and columns for subplots
            num_rows = 1
            num_cols = len(job_types)

            # Create subplots
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 5))

            # Loop over each job type and plot
            for i, job_type in enumerate(job_types):
                # Filter data for the current job type
                job_type_data = df[df['JobType'] == job_type]

                # Count plot of JobType
                sns.countplot(x='JobType', data=job_type_data, ax=axes[i])
                axes[i].set_xlabel('JobType')
                axes[i].set_ylabel('Count')
                axes[i].set_title('Distribution of ' + job_type)
                axes[i].tick_params(axis='x', rotation=45)

            plt.tight_layout()
            plt.show()
            ''', language='python'
        )
        jobtype_1()

        # Distribution of Job Type
        st.write('##### Distribution of Job Type')
        st.code(
            '''
            job_type_counts = df.JobType.value_counts()

            # Create a bar plot
            plt.bar(job_type_counts.index, job_type_counts.values)

            # Add labels and title
            plt.xlabel("Job Type")
            plt.ylabel("Count")
            plt.title("Distribution of Job Types")

            # Rotate the x-axis labels for better readability (optional)
            plt.xticks(rotation=45)

            plt.savefig('/Users/mac/Desktop/Streamlit App/Photo/jobtype-2.png')

            # Display the plot
            plt.show()
            ''', language='python'
        )
        jobtype_2()

        # Relationship between JobType and Min Salary
        st.write('##### Relationship between Job Type and Minimum Salary')
        st.code(
            '''
            grouped_data = df.groupby('JobType')['Salary_min'].mean()

            # Plotting a bar chart to visualize the relationship between 'JobType' and 'Salary_min'
            grouped_data.plot(kind='bar')
            plt.xlabel('JobType')
            plt.ylabel('Average Salary_min')
            plt.title('Relationship between JobType and Salary_min')
            plt.savefig('/Users/mac/Desktop/Streamlit App/Photo/jobtype-3.png')
            plt.show()
            ''', language='python'
        )
        jobtype_3()
EDA_JobType()

def EDA_PositionLevel():
    st.markdown("""<div id="pos-lvl"></div>""", unsafe_allow_html=True)
    st.subheader("5.2. Position Level")
    with st.expander("CLICK HERE TO SHOW DATA"):
        # Values Count
        st.write('##### Position Level Values Count')
        st.code(
            '''
            df.PositionLevel.value_counts()
            ''', language='python'
        )
        st.write(
            """
            Take a look at the result after checking.
            """
        )
        df = pd.read_csv('CSV/data2.csv')
        s = df.PositionLevel.value_counts()
        st.text(s)

        # Position Level
        st.write('##### Distribution of Position Levels')
        st.code(
            '''
            position_level_counts = df.PositionLevel.value_counts()

            # Set the style of the plot
            sns.set(style="whitegrid")

            # Create a bar plot using seaborn
            plt.figure(figsize=(10, 6))  # Set the figure size
            ax = sns.barplot(x=position_level_counts.index, y=position_level_counts.values)

            # Add labels and title
            plt.xlabel("Position Level", fontsize=12)
            plt.ylabel("Count", fontsize=12)
            plt.title("Distribution of Position Levels", fontsize=14)

            # Rotate the x-axis labels for better readability
            plt.xticks(rotation=45, ha='right')

            # Add value labels to the bars
            for p in ax.patches:
                ax.annotate(f"{p.get_height()}", (p.get_x() + p.get_width() / 2., p.get_height()),
                             ha='center', va='center', xytext=(0, 5), textcoords='offset points')

            # Display the plot
            plt.tight_layout()
            plt.savefig("/Users/mac/Desktop/Streamlit App/Photo/poslevel-1.png")
            plt.show()
            ''', language='python'
        )
        poslevel_1()

        # Position Level VS Salary
        st.write('##### Relationship between Position Level and Minimum Salary')
        st.code(
            '''
            # Grouping the data by 'PositionLevel' and calculating the mean of 'Salary_min' for each group
            grouped_data = df.groupby('PositionLevel')['Salary_min'].mean()

            # Plotting a bar chart to visualize the relationship between 'PositionLevel' and 'Salary_min'
            grouped_data.plot(kind='bar')
            plt.xlabel('PositionLevel')
            plt.ylabel('Average Salary_min')
            plt.title('Relationship between PositionLevel and Salary_min')
            plt.show()
            ''', language='python'
        )
        poslevel_vs_sal()
EDA_PositionLevel()

def EDA_Location():
    st.markdown("""<div id="loc"></div>""", unsafe_allow_html=True)
    st.subheader("5.3. Location")
    with st.expander("CLICK HERE TO SHOW DATA"):
        # Values Count
        st.write('##### Location Values Count')
        st.code(
            '''
            df.Location.value_counts()
            ''', language='python'
        )
        st.write(
            """
            Take a look at the result after checking.
            """
        )
        df = pd.read_csv('CSV/data2.csv')
        s = df.Location.value_counts()
        st.text(s)

        # Location and Frequency Count
        st.write('##### Location Frequency Count')
        st.code(
            '''
            from sklearn.cluster import KMeans

            # Assuming you have already loaded the DataFrame 'df' with the relevant data
            # Get the frequency counts of the 'Location' variable
            location_counts = df['Location'].value_counts()

            # Convert the location_counts into a DataFrame
            location_counts_df = pd.DataFrame({'Location': location_counts.index, 'Count': location_counts.values})

            # Perform K-Means clustering on the frequency counts
            kmeans = KMeans(n_clusters=3)  # You can adjust the number of clusters as desired
            kmeans.fit(location_counts_df[['Count']])

            # Get the cluster labels
            cluster_labels = kmeans.labels_

            # Add the cluster labels to the DataFrame
            location_counts_df['Cluster'] = cluster_labels

            # Assign unique colors to each cluster label
            cluster_colors = sns.color_palette('viridis', n_colors=len(location_counts_df['Cluster'].unique()))

            # Plotting a bar chart to visualize the relationship between 'Location' and the frequency counts with clusters
            plt.bar(location_counts_df['Location'], location_counts_df['Count'], color=[cluster_colors[i] for i in location_counts_df['Cluster']])
            plt.xlabel('Location')
            plt.ylabel('Frequency Count')
            plt.title('Relationship between Location and Frequency Count with Clusters')
            plt.xticks(rotation=90)
            plt.show()
            ''', language='python'
        )
        loc_freq_count()

        # Abbreviation Provinces
        st.write('##### Replace Provinces from full words to abbreviation')
        st.code(
            '''
            df_replaced = df.replace({
                'Phnom Penh': 'PP',
                'Preah Sihanouk': 'PS',
                'Battambang': 'BB',
                'Siem Reap': 'SR',
                'Bavet': 'BV',
                'Poipet': 'PPt',
                'Kampong Speu': 'KS',
                'Kampot': 'KPT',
                'Kandal': 'KDL',
                'Kampong Cham': 'KC',
                'Svay Rieng': 'SR',
                'Mondulkiri': 'MDK',
                'Banteay Meanchey': 'BMC',
                'Preah Vihear': 'PV',
                'Pursat': 'PS',
                'Kampong Chhnang': 'KCN',
                'Koh Kong': 'KK',
                'Cambodia': 'CM',
                'Kampong Thom': 'KT',
                'Tbong Khmum': 'TK'
            })
            ''', language='python'
        )

        # Replace location to short letters
        st.write('##### Plotting Location after replaced to abbreviation')
        st.code(
            '''
            location_counts = df_replaced.Location.value_counts()

            # Set the style of the plot
            sns.set(style="whitegrid")

            # Create a bar plot using seaborn
            plt.figure(figsize=(10, 6))  # Set the figure size
            ax = sns.barplot(x=location_counts.index, y=location_counts.values)

            # Add labels and title
            plt.xlabel("Location", fontsize=12)
            plt.ylabel("Count", fontsize=12)
            plt.title("Distribution of Locations", fontsize=14)

            # Rotate the x-axis labels for better readability
            plt.xticks(rotation=45, ha='right')

            # Add value labels to the bars
            for p in ax.patches:
                ax.annotate(f"{p.get_height()}", (p.get_x() + p.get_width() / 2., p.get_height()),
                             ha='center', va='center', xytext=(0, 5), textcoords='offset points')

            # Display the plot
            plt.tight_layout()
            plt.show()
            ''', language='python'
        )
        abbrev_loc()

        # Relationship between Salary_min and Location
        st.write('##### Relationship between Minimum Salary and Location')
        st.code(
            '''
            location_stats = filtered_df.groupby('Location')['Salary_min'].mean().reset_index()

            # Sort the locations by average salary in descending order
            location_stats = location_stats.sort_values('Salary_min', ascending=False)

            # Plotting a bar plot to visualize the relationship between 'Salary_min' and 'Location'
            plt.figure(figsize=(12, 6))
            sns.barplot(x='Location', y='Salary_min', data=location_stats, palette='viridis')
            plt.xlabel('Location')
            plt.ylabel('Average Salary_min')
            plt.title('Relationship between Salary_min and Location')
            plt.xticks(rotation=90)
            plt.show()
            ''', language='python'
        )
        sal_vs_loc()
EDA_Location()

def EDA_Work_Exp():
    st.markdown("""<div id="work-exp"></div>""", unsafe_allow_html=True)
    st.subheader("5.4. Working Experience")
    with st.expander("CLICK HERE TO SHOW DATA"):
        # Values Count
        st.write('##### Working Experience Values Count')
        st.code(
            '''
            df.WorkingExperience.value_counts()
            ''', language='python'
        )
        st.write(
            """
            Take a look at the result after checking.
            """
        )
        df = pd.read_csv('CSV/data2.csv')
        s = df.WorkingExperience.value_counts()
        st.text(s)

        st.write("##### Distribution of Working Experience")
        st.code(
            '''
            working_experience_counts = df.WorkingExperience.value_counts()
    
            # Set the style of the plot
            sns.set(style="whitegrid")
    
            # Create a bar plot using seaborn
            plt.figure(figsize=(8, 6))  # Set the figure size
            ax = sns.barplot(x=working_experience_counts.index, y=working_experience_counts.values)
    
            # Add labels and title
            plt.xlabel("Working Experience", fontsize=12)
            plt.ylabel("Count", fontsize=12)
            plt.title("Distribution of Working Experience", fontsize=14)
    
            # Rotate the x-axis labels for better readability
            plt.xticks(rotation=0)
    
            # Add value labels to the bars
            for p in ax.patches:
                ax.annotate(f"{p.get_height()}", (p.get_x() + p.get_width() / 2., p.get_height()),
                             ha='center', va='center', xytext=(0, 5), textcoords='offset points')
    
            # Display the plot
            plt.tight_layout()
            plt.show()
            ''', language='python'
        )
        work_exp()
EDA_Work_Exp()

def EDA_Qualification():
    st.markdown("""<div id="qualification"></div>""", unsafe_allow_html=True)
    st.subheader("5.5. Qualification")
    with st.expander("CLICK HERE TO SHOW DATA"):
        # Values Count
        st.write('##### Qualification Values Count')
        st.code(
            '''
            df.Qualification.value_counts()
            ''', language='python'
        )
        st.write(
            """
            Take a look at the result after checking.
            """
        )
        df = pd.read_csv('CSV/data2.csv')
        s = df.Qualification.value_counts()
        st.text(s)

        # Location and Frequency Count
        st.write('##### Distribution of Qualification')
        st.code(
            '''
            qualification_counts = df.Qualification.value_counts()

            # Set the style of the plot
            sns.set(style="whitegrid")

            # Create a bar plot using seaborn
            plt.figure(figsize=(10, 6))  # Set the figure size
            ax = sns.barplot(x=qualification_counts.index, y=qualification_counts.values)

            # Add labels and title
            plt.xlabel("Qualification", fontsize=12)
            plt.ylabel("Count", fontsize=12)
            plt.title("Distribution of Qualifications", fontsize=14)

            # Rotate the x-axis labels for better readability
            plt.xticks(rotation=45)

            # Add value labels to the bars
            for p in ax.patches:
                ax.annotate(f"{p.get_height()}", (p.get_x() + p.get_width() / 2., p.get_height()),
                             ha='center', va='center', xytext=(0, 5), textcoords='offset points')

            # Display the plot
            plt.tight_layout()
            plt.show()
            ''', language='python'
        )
        qualification_dis()

        # Relationship Between Qualification and Average Minimum Salary
        st.write('##### Relationship between Qualification and Average Minimum Salary')
        st.code(
            '''
            filtered_df = df.dropna(subset=['Salary_min', 'Qualification'])

            # Calculate the average salary for each qualification
            qualification_stats = filtered_df.groupby('Qualification')['Salary_min'].mean().reset_index()

            # Sort the qualifications by average salary in descending order
            qualification_stats = qualification_stats.sort_values('Salary_min', ascending=False)

            # Plotting a bar plot to visualize the relationship between 'Salary_min' and 'Qualification'
            plt.figure(figsize=(10, 6))
            sns.barplot(x='Qualification', y='Salary_min', data=qualification_stats, palette='viridis')
            plt.xlabel('Qualification')
            plt.ylabel('Average Salary_min')
            plt.title('Relationship between Qualification and Average Salary_min')
            plt.xticks(rotation=45)
            plt.show()
            })
            ''', language='python'
        )
        qualification_vs_sal()

        # Distribution of Qualification
        st.write('##### Distribution of Qualification')
        st.code(
            '''
            df_replaced = df.replace({
                'Bacc2': 'High School Graduate',
                'Associate': 'Associate Degree',
                'No': 'No Qualification',
                'Master': 'Master Degree',
                'Others': 'Other Qualification',
                'Diploma': 'Diploma',
                'Professional': 'Professional Degree',
                'Train': 'Trainee'
            })

            qualification_counts = df_replaced.Qualification.value_counts()

            # Create a bar plot
            qualification_counts.plot.bar(figsize=(10, 6))

            # Add labels and title
            plt.xlabel("Qualification", fontsize=12)
            plt.ylabel("Count", fontsize=12)
            plt.title("Distribution of Qualifications", fontsize=14)

            # Rotate the x-axis labels for better readability
            plt.xticks(rotation=45)

            # Display the plot
            plt.tight_layout()
            plt.show()
            })
            ''', language='python'
        )
        qualification_dis_after_abbre()
EDA_Qualification()

def EDA_Age():
    st.markdown("""<div id="age"></div>""", unsafe_allow_html=True)
    st.subheader("5.6. Age")
    with st.expander(label="CLICK HERE TO SHOW DATA", expanded=False):
        # Values Count
        st.write('##### Age Values Count')
        st.code(
            '''
            df.Age.value_counts()
            ''', language='python'
        )
        st.write(
            """
            Take a look at the result after checking.
            """
        )
        df = pd.read_csv('CSV/data2.csv')
        s = df.min_age.value_counts()
        st.text(s)

        # Age Distribution
        st.write('##### Distribution of Minimum Age')
        st.code(
            '''
            min_age_counts = df.min_age.value_counts()

            # Set the style of the plot
            sns.set(style="whitegrid")

            # Create a bar plot using seaborn
            plt.figure(figsize=(10, 6))  # Set the figure size
            ax = sns.barplot(x=min_age_counts.index, y=min_age_counts.values)

            # Add labels and title
            plt.xlabel("Minimum Age", fontsize=12)
            plt.ylabel("Count", fontsize=12)
            plt.title("Distribution of Minimum Ages", fontsize=14)

            # Rotate the x-axis labels for better readability
            plt.xticks(rotation=0)

            # Add value labels to the bars
            for p in ax.patches:
                ax.annotate(f"{p.get_height()}", (p.get_x() + p.get_width() / 2., p.get_height()),
                             ha='center', va='center', xytext=(0, 5), textcoords='offset points')

            # Display the plot
            plt.tight_layout()
            plt.show()
            ''', language='python'
        )
        age_dis()

        # Relationship between Minimum Age and Minimum Salary
        st.write('##### Relationship between Minimum Age and Minimum Salary')
        st.code(
            '''
            grouped_data = df.groupby('min_age')['Salary_min'].mean()

            # Plotting a bar chart to visualize the relationship between 'min_age' and 'Salary_min'
            grouped_data.plot(kind='bar')
            plt.xlabel('Minimum Age')
            plt.ylabel('Average Salary_min')
            plt.title('Relationship between Minimum Age and Salary_min')
            plt.show()
            ''', language='python'
        )
        age_vs_sal()
EDA_Age()

def EDA_Salary():
    st.markdown("""<div id="salary"></div>""", unsafe_allow_html=True)
    st.subheader("5.7. Salary")
    with st.expander("CLICK HERE TO SHOW DATA"):
        # Values Count
        st.write('##### Salary Values Count')
        st.code(
            '''
            df.Salary_min.value_counts()
            ''', language='python'
        )
        st.write(
            """
            Take a look at the result after checking.
            """
        )
        df = pd.read_csv('CSV/data2.csv')
        s = df.Salary_min.value_counts()
        st.text(s)

        st.write('##### Distribution of Minimum Salaries')
        st.code(
            '''
            salary_min_counts = df.Salary_min.value_counts()

            # Set the style of the plot
            sns.set(style="whitegrid")

            # Create a bar plot using seaborn
            plt.figure(figsize=(10, 6))  # Set the figure size
            ax = sns.barplot(x=salary_min_counts.index, y=salary_min_counts.values)

            # Add labels and title
            plt.xlabel("Minimum Salary", fontsize=12)
            plt.ylabel("Count", fontsize=12)
            plt.title("Distribution of Minimum Salaries", fontsize=14)

            # Rotate the x-axis labels for better readability
            plt.xticks(rotation=0)

            # Add value labels to the bars
            for p in ax.patches:
                ax.annotate(f"{p.get_height()}", (p.get_x() + p.get_width() / 2., p.get_height()),
                             ha='center', va='center', xytext=(0, 5), textcoords='offset points')

            # Display the plot
            plt.tight_layout()
            plt.show()
            ''', language='python'
        )
        sal_dis()

        st.write('##### Pairwise Scatter Plot')
        st.code(
            '''
            # Pairwise scatter plot for numeric variables
            numeric_vars = ['min_age', 'WorkingExperience']
            sns.pairplot(df, x_vars=numeric_vars, y_vars='Salary_min', height=4)
            plt.show()
            ''', language='python'
        )
        pairwise_sal()

        st.write('##### Relationship between Minimum Age and Minimum Salary')
        st.code(
            '''
            sns.scatterplot(x=df['min_age'], y=df['Salary_min'])
            plt.xlabel('Minimum Age')
            plt.ylabel('Salary_min')
            plt.title('Relationship between Minimum Age and Salary_min')
            plt.show()
            ''', language='python'
        )
        pairwise_age_sal()

        st.write('##### Subplots with Minimum Salary')
        st.code(
            '''
            plt.figure(figsize=(12, 8))

            # JobType vs. Salary_min
            plt.subplot(2, 3, 1)
            sns.barplot(x='JobType', y='Salary_min', data=df)
            plt.xlabel('JobType')
            plt.ylabel('Salary_min')
            plt.xticks(rotation=90)

            # PositionLevel vs. Salary_min
            plt.subplot(2, 3, 2)
            sns.barplot(x='PositionLevel', y='Salary_min', data=df)
            plt.xlabel('PositionLevel')
            plt.ylabel('Salary_min')
            plt.xticks(rotation=90)

            # Location vs. Salary_min
            plt.subplot(2, 3, 3)
            sns.barplot(x='Location', y='Salary_min', data=df)
            plt.xlabel('Location')
            plt.ylabel('Salary_min')
            plt.xticks(rotation=90)

            # WorkingExperience vs. Salary_min
            plt.subplot(2, 3, 4)
            sns.barplot(x='WorkingExperience', y='Salary_min', data=df)
            plt.xlabel('WorkingExperience')
            plt.ylabel('Salary_min')

            # Qualification vs. Salary_min
            plt.subplot(2, 3, 5)
            sns.barplot(x='Qualification', y='Salary_min', data=df)
            plt.xlabel('Qualification')
            plt.ylabel('Salary_min')
            plt.xticks(rotation=90)

            # min_age vs. Salary_min
            plt.subplot(2, 3, 6)
            sns.barplot(x='min_age', y='Salary_min', data=df)
            plt.xlabel('min_age')
            plt.ylabel('Salary_min')
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.show()
            ''', language='python'
        )
        subplots_sal()
EDA_Salary()

# Show pdf: report and guideline
def show_pdf(pdf_file):
    with open(pdf_file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf">'
    st.markdown(pdf_display, unsafe_allow_html=True)

st.markdown("""<div id="report"></div>""", unsafe_allow_html=True)
st.markdown("<div style='font-size: 30px';><center><b>Job Analysis Report</b></center></div>", unsafe_allow_html=True)
show_pdf("PDF Report/Report.pdf")

st.markdown("""<div id="guideline"></div>""", unsafe_allow_html=True)
st.markdown("<div style='font-size: 30px';><center><b>Job Analysis Guideline</b></center></div>", unsafe_allow_html=True)
show_pdf("PDF Report/Guideline.pdf")
