import streamlit as st
import pandas as pd
from pathlib import Path
import base64
import io
import os
import tempfile
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.feature_selection import SelectKBest, chi2
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import itertools
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
# from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns

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

    st.sidebar.write("[6. Feature Engineering](#feat-engineering)")
    st.sidebar.write("[6.1. Pre-Processing](#pre-processing)")

    st.sidebar.write("[7. Model Building](#model-building)")
    st.sidebar.write("[7.1. Train Split Test](#train_split)")
    st.sidebar.write("[7.2. Classification](#classification)")
    st.sidebar.write("[7.3. Accuracy Evaluation](#accuracy-evaluation)")

    st.sidebar.write("[8. Feature Selection](#feat-selection)")

    st.sidebar.write("[9. SVM (Support Vector Machine) with Scikit-learn](#svm)")

    st.sidebar.write("[10. Gradient Boosting Classifier](#gbc)")

    st.sidebar.write("[11. Result](#result)")

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


def confusion():
    con_mat = "<center><figure><img src='data:image/png;base64,{}' class='img-fluid' width=650><figcaption></figcaption></figure></center>".format(
        img_to_bytes("Photo/confusion_matrices.png")
    )
    container = st.container()
    with container:
        st.markdown(con_mat, unsafe_allow_html=True)


def feat_importance():
    f_imp = "<center><figure><img src='data:image/png;base64,{}' class='img-fluid' width=650><figcaption>Features Importances</figcaption></figure></center>".format(
        img_to_bytes("Photo/feat_imp.png")
    )
    container = st.container()
    with container:
        st.markdown(f_imp, unsafe_allow_html=True)


def image():
    itc_path = load_data("Photo/itc.png")
    header_html_1 = "<a href='https://itc.edu.kh/'><center><figure><img src='data:image/png;base64,{}' class='img-fluid' width=250><figcaption>Institute of technology of Cambodia</figcaption></figure></center></a>".format(
        img_to_bytes(itc_path)
    )
    ams_path = load_data("Photo/dep.jpg")
    header_html_2 = "<a href='https://itc.edu.kh/home-ams/'><center><figure><img src='data:image/png;base64,{}' class='img-fluid' width=246><figcaption>Department of Applied Mathematics and Statistics</figcaption></figure></center></a>".format(
        img_to_bytes(ams_path)
    )
    ministry_path = load_data("Photo/moey.png")
    header_html_3 = "<a href='http://moeys.gov.kh/'><center><figure><img src='data:image/png;base64,{}' class='img-fluid' width=183><figcaption>Ministry of Education, Youth and Sport (Cambodia)</figcaption></figure></center></a>".format(
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
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
            ["SENG Lay", "VANNAK Vireakyuth", "YA Manon", "VANN Visal", "TAING Kimmeng", "VINLAY Anusar"])
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

df = pd.read_csv('CSV/data2.csv')
def data_exploration_4():
    ### Data Exploration ###
    st.write("## `LET'S DIVE INTO OUR PROJECT!`")
    st.markdown("""<div id="data-exploration"></div>""", unsafe_allow_html=True)
    st.header("4. Data Exploration: Exploring the Dataset")
    with st.expander("CLICK HERE TO SHOW DATA"):
        # Import libraries
        st.markdown("""<div id="import-dataset"></div>""", unsafe_allow_html=True)
        st.subheader('4.1. Importing Python Language Libraries')
        st.write(
            """
            Bonjour, First of all we need to import important libraries needed for our project first! <3
            """
        )
        code = '''
            import streamlit as st
            import pandas as pd
            from pathlib import Path
            import base64
            import io
            import os
            import tempfile
            import numpy as np
            from sklearn.model_selection import train_test_split
            from sklearn import preprocessing
            from sklearn.neighbors import KNeighborsClassifier
            from sklearn import metrics
            from sklearn.feature_selection import SelectKBest, chi2
            from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
            import itertools
            import matplotlib.pyplot as plt
            from sklearn.feature_selection import RFE
            from sklearn.ensemble import GradientBoostingClassifier
            from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
            from sklearn.ensemble import GradientBoostingClassifier
            from sklearn.metrics import accuracy_score
            import seaborn as sns
            '''
        st.code(code, language='python')

        # Load dataset
        st.markdown("""<div id="origin-dataset"></div>""", unsafe_allow_html=True)
        st.subheader('4.2. The Original Dataset `AllPhnomList.csv` File')
        # with st.expander("CLICK HERE TO SHOW DATA"):
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
        st.dataframe(df)

        # Describe the whole dataset
        st.markdown("""<div id="data-description"></div>""", unsafe_allow_html=True)
        st.subheader('4.3. Data Description')
        # with st.expander("CLICK HERE TO SHOW DATA"):
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
        st.dataframe(df.describe())

        # Dataset Info
        st.markdown("""<div id="info-data"></div>""", unsafe_allow_html=True)
        st.subheader('4.3.1. Dataset Info')
        # with st.expander("CLICK HERE TO SHOW DATA"):
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
        # with st.expander("CLICK HERE TO SHOW DATA"):
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
        # with st.expander("CLICK HERE TO SHOW DATA"):
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


# data_exploration_4()
#
# def data_cleaning():
        st.markdown("""<div id="data-cleaning"></div>""", unsafe_allow_html=True)
        st.subheader('4.4. Data Cleaning')

        ### 4.4.1
        st.markdown("""<div id="missing-vals"></div>""", unsafe_allow_html=True)
        st.subheader('4.4.1. Missing Values')
        # with st.expander("CLICK HERE TO SHOW DATA"):
        st.code(
            '''
            # Drop unnescessaries features and cleaning 
            df.drop(columns=['Salary_max', 'max_age'], inplace=True, axis =1)
            
            # Check Missing Values
            df.isnull().sum()
            ''', language='python'
        )
        df.drop(columns=['Salary_max', 'max_age'], inplace=True, axis=1)

        st.write(
            """
            Take a look at the result after checking.
            """
        )
        st.text(df.isnull().sum())

        ### 4.4.2
        st.markdown("""<div id="dup-vals"></div>""", unsafe_allow_html=True)
        st.subheader('4.4.2. Duplicated Values')
        # with st.expander("CLICK HERE TO SHOW DATA"):
        st.code(
            '''
            # Sum up all duplicated values
            df.duplicated().sum()
            ''', language='python'
        )

        st.text(f'Duplicated values: {df.duplicated().sum()}')

        st.code(
            '''
            # Drop all duplicated values
            df.drop_duplicates(inplace= True)
            ''', language='python'
            )
        df.drop_duplicates(inplace=True)

        ### 4.4.3
        st.markdown("""<div id="outlier"></div>""", unsafe_allow_html=True)
        st.subheader('4.4.3. Outlier')
        # with st.expander("CLICK HERE TO SHOW DATA"):
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
# data_cleaning()
data_exploration_4()


def EDA():
    st.markdown("""<div id="eda"></div>""", unsafe_allow_html=True)
    st.header("5. Exploratory Data Analysis (EDA)")
    st.write('''
    In this EDA section, we're going to check details on each variables in our dataset.
    ''')
    with st.expander("CLICK HERE TO SHOW DATA"):
        st.markdown("""<div id="job-type"></div>""", unsafe_allow_html=True)
        st.subheader("5.1. Job Type")
        # with st.expander("CLICK HERE TO SHOW DATA"):
        st.write('##### Job Type Values Counts')
        # Values Count
        st.code(
            '''
            df.JobType.value_counts()
            ''', language='python'
        )
        st.write(
            """
            Take a look at the result after checking.
            """
        )
        st.text(df.JobType.value_counts())

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
        st.markdown("""<div id="pos-lvl"></div>""", unsafe_allow_html=True)
        st.subheader("5.2. Position Level")
        # with st.expander("CLICK HERE TO SHOW DATA"):
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
        st.text(df.PositionLevel.value_counts())

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

        st.markdown("""<div id="loc"></div>""", unsafe_allow_html=True)
        st.subheader("5.3. Location")
        # with st.expander("CLICK HERE TO SHOW DATA"):
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
        st.text(df.Location.value_counts())

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

        st.markdown("""<div id="work-exp"></div>""", unsafe_allow_html=True)
        st.subheader("5.4. Working Experience")
        # with st.expander("CLICK HERE TO SHOW DATA"):
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
        st.text(df.WorkingExperience.value_counts())

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

        st.markdown("""<div id="qualification"></div>""", unsafe_allow_html=True)
        st.subheader("5.5. Qualification")
        # with st.expander("CLICK HERE TO SHOW DATA"):
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
        st.text(df.Qualification.value_counts())

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
        st.markdown("""<div id="age"></div>""", unsafe_allow_html=True)
        st.subheader("5.6. Age")
    # with st.expander(label="CLICK HERE TO SHOW DATA", expanded=False):
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
        st.text(df.min_age.value_counts())

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
        st.markdown("""<div id="salary"></div>""", unsafe_allow_html=True)
        st.subheader("5.7. Salary")
        # with st.expander("CLICK HERE TO SHOW DATA"):
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
        st.text(df.Salary_min.value_counts())

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

EDA()

# def EDA_PositionLevel():
    # st.markdown("""<div id="pos-lvl"></div>""", unsafe_allow_html=True)
    # st.subheader("5.2. Position Level")
    # with st.expander("CLICK HERE TO SHOW DATA"):
    #     # Values Count
    #     st.write('##### Position Level Values Count')
    #     st.code(
    #         '''
    #         df.PositionLevel.value_counts()
    #         ''', language='python'
    #     )
    #     st.write(
    #         """
    #         Take a look at the result after checking.
    #         """
    #     )
    #     st.text(df.PositionLevel.value_counts())
    #
    #     # Position Level
    #     st.write('##### Distribution of Position Levels')
    #     st.code(
    #         '''
    #         position_level_counts = df.PositionLevel.value_counts()
    #
    #         # Set the style of the plot
    #         sns.set(style="whitegrid")
    #
    #         # Create a bar plot using seaborn
    #         plt.figure(figsize=(10, 6))  # Set the figure size
    #         ax = sns.barplot(x=position_level_counts.index, y=position_level_counts.values)
    #
    #         # Add labels and title
    #         plt.xlabel("Position Level", fontsize=12)
    #         plt.ylabel("Count", fontsize=12)
    #         plt.title("Distribution of Position Levels", fontsize=14)
    #
    #         # Rotate the x-axis labels for better readability
    #         plt.xticks(rotation=45, ha='right')
    #
    #         # Add value labels to the bars
    #         for p in ax.patches:
    #             ax.annotate(f"{p.get_height()}", (p.get_x() + p.get_width() / 2., p.get_height()),
    #                          ha='center', va='center', xytext=(0, 5), textcoords='offset points')
    #
    #         # Display the plot
    #         plt.tight_layout()
    #         plt.savefig("/Users/mac/Desktop/Streamlit App/Photo/poslevel-1.png")
    #         plt.show()
    #         ''', language='python'
    #     )
    #     poslevel_1()
    #
    #     # Position Level VS Salary
    #     st.write('##### Relationship between Position Level and Minimum Salary')
    #     st.code(
    #         '''
    #         # Grouping the data by 'PositionLevel' and calculating the mean of 'Salary_min' for each group
    #         grouped_data = df.groupby('PositionLevel')['Salary_min'].mean()
    #
    #         # Plotting a bar chart to visualize the relationship between 'PositionLevel' and 'Salary_min'
    #         grouped_data.plot(kind='bar')
    #         plt.xlabel('PositionLevel')
    #         plt.ylabel('Average Salary_min')
    #         plt.title('Relationship between PositionLevel and Salary_min')
    #         plt.show()
    #         ''', language='python'
    #     )
    #     poslevel_vs_sal()
# EDA_PositionLevel()

# def EDA_Location():
    # st.markdown("""<div id="loc"></div>""", unsafe_allow_html=True)
    # st.subheader("5.3. Location")
    # with st.expander("CLICK HERE TO SHOW DATA"):
    #     # Values Count
    #     st.write('##### Location Values Count')
    #     st.code(
    #         '''
    #         df.Location.value_counts()
    #         ''', language='python'
    #     )
    #     st.write(
    #         """
    #         Take a look at the result after checking.
    #         """
    #     )
    #     st.text(df.Location.value_counts())
    #
    #     # Location and Frequency Count
    #     st.write('##### Location Frequency Count')
    #     st.code(
    #         '''
    #         from sklearn.cluster import KMeans
    #
    #         # Assuming you have already loaded the DataFrame 'df' with the relevant data
    #         # Get the frequency counts of the 'Location' variable
    #         location_counts = df['Location'].value_counts()
    #
    #         # Convert the location_counts into a DataFrame
    #         location_counts_df = pd.DataFrame({'Location': location_counts.index, 'Count': location_counts.values})
    #
    #         # Perform K-Means clustering on the frequency counts
    #         kmeans = KMeans(n_clusters=3)  # You can adjust the number of clusters as desired
    #         kmeans.fit(location_counts_df[['Count']])
    #
    #         # Get the cluster labels
    #         cluster_labels = kmeans.labels_
    #
    #         # Add the cluster labels to the DataFrame
    #         location_counts_df['Cluster'] = cluster_labels
    #
    #         # Assign unique colors to each cluster label
    #         cluster_colors = sns.color_palette('viridis', n_colors=len(location_counts_df['Cluster'].unique()))
    #
    #         # Plotting a bar chart to visualize the relationship between 'Location' and the frequency counts with clusters
    #         plt.bar(location_counts_df['Location'], location_counts_df['Count'], color=[cluster_colors[i] for i in location_counts_df['Cluster']])
    #         plt.xlabel('Location')
    #         plt.ylabel('Frequency Count')
    #         plt.title('Relationship between Location and Frequency Count with Clusters')
    #         plt.xticks(rotation=90)
    #         plt.show()
    #         ''', language='python'
    #     )
    #     loc_freq_count()
    #
    #     # Abbreviation Provinces
    #     st.write('##### Replace Provinces from full words to abbreviation')
    #     st.code(
    #         '''
    #         df_replaced = df.replace({
    #             'Phnom Penh': 'PP',
    #             'Preah Sihanouk': 'PS',
    #             'Battambang': 'BB',
    #             'Siem Reap': 'SR',
    #             'Bavet': 'BV',
    #             'Poipet': 'PPt',
    #             'Kampong Speu': 'KS',
    #             'Kampot': 'KPT',
    #             'Kandal': 'KDL',
    #             'Kampong Cham': 'KC',
    #             'Svay Rieng': 'SR',
    #             'Mondulkiri': 'MDK',
    #             'Banteay Meanchey': 'BMC',
    #             'Preah Vihear': 'PV',
    #             'Pursat': 'PS',
    #             'Kampong Chhnang': 'KCN',
    #             'Koh Kong': 'KK',
    #             'Cambodia': 'CM',
    #             'Kampong Thom': 'KT',
    #             'Tbong Khmum': 'TK'
    #         })
    #         ''', language='python'
    #     )
    #     df_replaced = df.replace({
    #         'Phnom Penh': 'PP',
    #         'Preah Sihanouk': 'PS',
    #         'Battambang': 'BB',
    #         'Siem Reap': 'SR',
    #         'Bavet': 'BV',
    #         'Poipet': 'PPt',
    #         'Kampong Speu': 'KS',
    #         'Kampot': 'KPT',
    #         'Kandal': 'KDL',
    #         'Kampong Cham': 'KC',
    #         'Svay Rieng': 'SR',
    #         'Mondulkiri': 'MDK',
    #         'Banteay Meanchey': 'BMC',
    #         'Preah Vihear': 'PV',
    #         'Pursat': 'PS',
    #         'Kampong Chhnang': 'KCN',
    #         'Koh Kong': 'KK',
    #         'Cambodia': 'CM',
    #         'Kampong Thom': 'KT',
    #         'Tbong Khmum': 'TK'
    #     })
    #
    #     # Replace location to short letters
    #     st.write('##### Plotting Location after replaced to abbreviation')
    #     st.code(
    #         '''
    #         location_counts = df_replaced.Location.value_counts()
    #
    #         # Set the style of the plot
    #         sns.set(style="whitegrid")
    #
    #         # Create a bar plot using seaborn
    #         plt.figure(figsize=(10, 6))  # Set the figure size
    #         ax = sns.barplot(x=location_counts.index, y=location_counts.values)
    #
    #         # Add labels and title
    #         plt.xlabel("Location", fontsize=12)
    #         plt.ylabel("Count", fontsize=12)
    #         plt.title("Distribution of Locations", fontsize=14)
    #
    #         # Rotate the x-axis labels for better readability
    #         plt.xticks(rotation=45, ha='right')
    #
    #         # Add value labels to the bars
    #         for p in ax.patches:
    #             ax.annotate(f"{p.get_height()}", (p.get_x() + p.get_width() / 2., p.get_height()),
    #                          ha='center', va='center', xytext=(0, 5), textcoords='offset points')
    #
    #         # Display the plot
    #         plt.tight_layout()
    #         plt.show()
    #         ''', language='python'
    #     )
    #     abbrev_loc()
    #
    #     # Relationship between Salary_min and Location
    #     st.write('##### Relationship between Minimum Salary and Location')
    #     st.code(
    #         '''
    #         location_stats = filtered_df.groupby('Location')['Salary_min'].mean().reset_index()
    #
    #         # Sort the locations by average salary in descending order
    #         location_stats = location_stats.sort_values('Salary_min', ascending=False)
    #
    #         # Plotting a bar plot to visualize the relationship between 'Salary_min' and 'Location'
    #         plt.figure(figsize=(12, 6))
    #         sns.barplot(x='Location', y='Salary_min', data=location_stats, palette='viridis')
    #         plt.xlabel('Location')
    #         plt.ylabel('Average Salary_min')
    #         plt.title('Relationship between Salary_min and Location')
    #         plt.xticks(rotation=90)
    #         plt.show()
    #         ''', language='python'
    #     )
    #     sal_vs_loc()


# EDA_Location()


# def EDA_Work_Exp():
    # st.markdown("""<div id="work-exp"></div>""", unsafe_allow_html=True)
    # st.subheader("5.4. Working Experience")
    # with st.expander("CLICK HERE TO SHOW DATA"):
    #     # Values Count
    #     st.write('##### Working Experience Values Count')
    #     st.code(
    #         '''
    #         df.WorkingExperience.value_counts()
    #         ''', language='python'
    #     )
    #     st.write(
    #         """
    #         Take a look at the result after checking.
    #         """
    #     )
    #     st.text(df.WorkingExperience.value_counts())
    #
    #     st.write("##### Distribution of Working Experience")
    #     st.code(
    #         '''
    #         working_experience_counts = df.WorkingExperience.value_counts()
    #
    #         # Set the style of the plot
    #         sns.set(style="whitegrid")
    #
    #         # Create a bar plot using seaborn
    #         plt.figure(figsize=(8, 6))  # Set the figure size
    #         ax = sns.barplot(x=working_experience_counts.index, y=working_experience_counts.values)
    #
    #         # Add labels and title
    #         plt.xlabel("Working Experience", fontsize=12)
    #         plt.ylabel("Count", fontsize=12)
    #         plt.title("Distribution of Working Experience", fontsize=14)
    #
    #         # Rotate the x-axis labels for better readability
    #         plt.xticks(rotation=0)
    #
    #         # Add value labels to the bars
    #         for p in ax.patches:
    #             ax.annotate(f"{p.get_height()}", (p.get_x() + p.get_width() / 2., p.get_height()),
    #                          ha='center', va='center', xytext=(0, 5), textcoords='offset points')
    #
    #         # Display the plot
    #         plt.tight_layout()
    #         plt.show()
    #         ''', language='python'
    #     )
    #     work_exp()


# EDA_Work_Exp()


# def EDA_Qualification():
    # st.markdown("""<div id="qualification"></div>""", unsafe_allow_html=True)
    # st.subheader("5.5. Qualification")
    # with st.expander("CLICK HERE TO SHOW DATA"):
    #     # Values Count
    #     st.write('##### Qualification Values Count')
    #     st.code(
    #         '''
    #         df.Qualification.value_counts()
    #         ''', language='python'
    #     )
    #     st.write(
    #         """
    #         Take a look at the result after checking.
    #         """
    #     )
    #     st.text(df.Qualification.value_counts())
    #
    #     # Location and Frequency Count
    #     st.write('##### Distribution of Qualification')
    #     st.code(
    #         '''
    #         qualification_counts = df.Qualification.value_counts()
    #
    #         # Set the style of the plot
    #         sns.set(style="whitegrid")
    #
    #         # Create a bar plot using seaborn
    #         plt.figure(figsize=(10, 6))  # Set the figure size
    #         ax = sns.barplot(x=qualification_counts.index, y=qualification_counts.values)
    #
    #         # Add labels and title
    #         plt.xlabel("Qualification", fontsize=12)
    #         plt.ylabel("Count", fontsize=12)
    #         plt.title("Distribution of Qualifications", fontsize=14)
    #
    #         # Rotate the x-axis labels for better readability
    #         plt.xticks(rotation=45)
    #
    #         # Add value labels to the bars
    #         for p in ax.patches:
    #             ax.annotate(f"{p.get_height()}", (p.get_x() + p.get_width() / 2., p.get_height()),
    #                          ha='center', va='center', xytext=(0, 5), textcoords='offset points')
    #
    #         # Display the plot
    #         plt.tight_layout()
    #         plt.show()
    #         ''', language='python'
    #     )
    #     qualification_dis()
    #
    #     # Relationship Between Qualification and Average Minimum Salary
    #     st.write('##### Relationship between Qualification and Average Minimum Salary')
    #     st.code(
    #         '''
    #         filtered_df = df.dropna(subset=['Salary_min', 'Qualification'])
    #
    #         # Calculate the average salary for each qualification
    #         qualification_stats = filtered_df.groupby('Qualification')['Salary_min'].mean().reset_index()
    #
    #         # Sort the qualifications by average salary in descending order
    #         qualification_stats = qualification_stats.sort_values('Salary_min', ascending=False)
    #
    #         # Plotting a bar plot to visualize the relationship between 'Salary_min' and 'Qualification'
    #         plt.figure(figsize=(10, 6))
    #         sns.barplot(x='Qualification', y='Salary_min', data=qualification_stats, palette='viridis')
    #         plt.xlabel('Qualification')
    #         plt.ylabel('Average Salary_min')
    #         plt.title('Relationship between Qualification and Average Salary_min')
    #         plt.xticks(rotation=45)
    #         plt.show()
    #         })
    #         ''', language='python'
    #     )
    #     qualification_vs_sal()
    #
    #     # Distribution of Qualification
    #     st.write('##### Distribution of Qualification')
    #     st.code(
    #         '''
    #         df_replaced = df.replace({
    #             'Bacc2': 'High School Graduate',
    #             'Associate': 'Associate Degree',
    #             'No': 'No Qualification',
    #             'Master': 'Master Degree',
    #             'Others': 'Other Qualification',
    #             'Diploma': 'Diploma',
    #             'Professional': 'Professional Degree',
    #             'Train': 'Trainee'
    #         })
    #
    #         qualification_counts = df_replaced.Qualification.value_counts()
    #
    #         # Create a bar plot
    #         qualification_counts.plot.bar(figsize=(10, 6))
    #
    #         # Add labels and title
    #         plt.xlabel("Qualification", fontsize=12)
    #         plt.ylabel("Count", fontsize=12)
    #         plt.title("Distribution of Qualifications", fontsize=14)
    #
    #         # Rotate the x-axis labels for better readability
    #         plt.xticks(rotation=45)
    #
    #         # Display the plot
    #         plt.tight_layout()
    #         plt.show()
    #         })
    #         ''', language='python'
    #     )
    #     qualification_dis_after_abbre()


# EDA_Qualification()


# def EDA_Age():
    # st.markdown("""<div id="age"></div>""", unsafe_allow_html=True)
    # st.subheader("5.6. Age")
    # with st.expander(label="CLICK HERE TO SHOW DATA", expanded=False):
    #     # Values Count
    #     st.write('##### Age Values Count')
    #     st.code(
    #         '''
    #         df.Age.value_counts()
    #         ''', language='python'
    #     )
    #     st.write(
    #         """
    #         Take a look at the result after checking.
    #         """
    #     )
    #     st.text(df.min_age.value_counts())
    #
    #     # Age Distribution
    #     st.write('##### Distribution of Minimum Age')
    #     st.code(
    #         '''
    #         min_age_counts = df.min_age.value_counts()
    #
    #         # Set the style of the plot
    #         sns.set(style="whitegrid")
    #
    #         # Create a bar plot using seaborn
    #         plt.figure(figsize=(10, 6))  # Set the figure size
    #         ax = sns.barplot(x=min_age_counts.index, y=min_age_counts.values)
    #
    #         # Add labels and title
    #         plt.xlabel("Minimum Age", fontsize=12)
    #         plt.ylabel("Count", fontsize=12)
    #         plt.title("Distribution of Minimum Ages", fontsize=14)
    #
    #         # Rotate the x-axis labels for better readability
    #         plt.xticks(rotation=0)
    #
    #         # Add value labels to the bars
    #         for p in ax.patches:
    #             ax.annotate(f"{p.get_height()}", (p.get_x() + p.get_width() / 2., p.get_height()),
    #                          ha='center', va='center', xytext=(0, 5), textcoords='offset points')
    #
    #         # Display the plot
    #         plt.tight_layout()
    #         plt.show()
    #         ''', language='python'
    #     )
    #     age_dis()
    #
    #     # Relationship between Minimum Age and Minimum Salary
    #     st.write('##### Relationship between Minimum Age and Minimum Salary')
    #     st.code(
    #         '''
    #         grouped_data = df.groupby('min_age')['Salary_min'].mean()
    #
    #         # Plotting a bar chart to visualize the relationship between 'min_age' and 'Salary_min'
    #         grouped_data.plot(kind='bar')
    #         plt.xlabel('Minimum Age')
    #         plt.ylabel('Average Salary_min')
    #         plt.title('Relationship between Minimum Age and Salary_min')
    #         plt.show()
    #         ''', language='python'
    #     )
    #     age_vs_sal()


# EDA_Age()


# def EDA_Salary():
    # st.markdown("""<div id="salary"></div>""", unsafe_allow_html=True)
    # st.subheader("5.7. Salary")
    # with st.expander("CLICK HERE TO SHOW DATA"):
    #     # Values Count
    #     st.write('##### Salary Values Count')
    #     st.code(
    #         '''
    #         df.Salary_min.value_counts()
    #         ''', language='python'
    #     )
    #     st.write(
    #         """
    #         Take a look at the result after checking.
    #         """
    #     )
    #     st.text(df.Salary_min.value_counts())
    #
    #     st.write('##### Distribution of Minimum Salaries')
    #     st.code(
    #         '''
    #         salary_min_counts = df.Salary_min.value_counts()
    #
    #         # Set the style of the plot
    #         sns.set(style="whitegrid")
    #
    #         # Create a bar plot using seaborn
    #         plt.figure(figsize=(10, 6))  # Set the figure size
    #         ax = sns.barplot(x=salary_min_counts.index, y=salary_min_counts.values)
    #
    #         # Add labels and title
    #         plt.xlabel("Minimum Salary", fontsize=12)
    #         plt.ylabel("Count", fontsize=12)
    #         plt.title("Distribution of Minimum Salaries", fontsize=14)
    #
    #         # Rotate the x-axis labels for better readability
    #         plt.xticks(rotation=0)
    #
    #         # Add value labels to the bars
    #         for p in ax.patches:
    #             ax.annotate(f"{p.get_height()}", (p.get_x() + p.get_width() / 2., p.get_height()),
    #                          ha='center', va='center', xytext=(0, 5), textcoords='offset points')
    #
    #         # Display the plot
    #         plt.tight_layout()
    #         plt.show()
    #         ''', language='python'
    #     )
    #     sal_dis()
    #
    #     st.write('##### Pairwise Scatter Plot')
    #     st.code(
    #         '''
    #         # Pairwise scatter plot for numeric variables
    #         numeric_vars = ['min_age', 'WorkingExperience']
    #         sns.pairplot(df, x_vars=numeric_vars, y_vars='Salary_min', height=4)
    #         plt.show()
    #         ''', language='python'
    #     )
    #     pairwise_sal()
    #
    #     st.write('##### Relationship between Minimum Age and Minimum Salary')
    #     st.code(
    #         '''
    #         sns.scatterplot(x=df['min_age'], y=df['Salary_min'])
    #         plt.xlabel('Minimum Age')
    #         plt.ylabel('Salary_min')
    #         plt.title('Relationship between Minimum Age and Salary_min')
    #         plt.show()
    #         ''', language='python'
    #     )
    #     pairwise_age_sal()
    #
    #     st.write('##### Subplots with Minimum Salary')
    #     st.code(
    #         '''
    #         plt.figure(figsize=(12, 8))
    #
    #         # JobType vs. Salary_min
    #         plt.subplot(2, 3, 1)
    #         sns.barplot(x='JobType', y='Salary_min', data=df)
    #         plt.xlabel('JobType')
    #         plt.ylabel('Salary_min')
    #         plt.xticks(rotation=90)
    #
    #         # PositionLevel vs. Salary_min
    #         plt.subplot(2, 3, 2)
    #         sns.barplot(x='PositionLevel', y='Salary_min', data=df)
    #         plt.xlabel('PositionLevel')
    #         plt.ylabel('Salary_min')
    #         plt.xticks(rotation=90)
    #
    #         # Location vs. Salary_min
    #         plt.subplot(2, 3, 3)
    #         sns.barplot(x='Location', y='Salary_min', data=df)
    #         plt.xlabel('Location')
    #         plt.ylabel('Salary_min')
    #         plt.xticks(rotation=90)
    #
    #         # WorkingExperience vs. Salary_min
    #         plt.subplot(2, 3, 4)
    #         sns.barplot(x='WorkingExperience', y='Salary_min', data=df)
    #         plt.xlabel('WorkingExperience')
    #         plt.ylabel('Salary_min')
    #
    #         # Qualification vs. Salary_min
    #         plt.subplot(2, 3, 5)
    #         sns.barplot(x='Qualification', y='Salary_min', data=df)
    #         plt.xlabel('Qualification')
    #         plt.ylabel('Salary_min')
    #         plt.xticks(rotation=90)
    #
    #         # min_age vs. Salary_min
    #         plt.subplot(2, 3, 6)
    #         sns.barplot(x='min_age', y='Salary_min', data=df)
    #         plt.xlabel('min_age')
    #         plt.ylabel('Salary_min')
    #         plt.xticks(rotation=90)
    #         plt.tight_layout()
    #         plt.show()
    #         ''', language='python'
    #     )
    #     subplots_sal()
# EDA_Salary()

def feature_engineering():
    st.markdown("""<div id="feat-engineering"></div>""", unsafe_allow_html=True)
    st.header("6. Feature Engineering")
    with st.expander("CLICK HERE TO SHOW DATA"):
        st.markdown("""<div id="pre-processing"></div>""", unsafe_allow_html=True)
        st.subheader("6.1. Pre-Processing")
        st.write('##### Qualification')
        st.code(
                '''
                # Define the ranges for qualification
                ranges = {
                    'No': 0,
                    'Others':0,
                    'Diploma': 1,
                    'Associate': 3,
                    'Bacc2': 2,
                    'Bachelor': 4,
                    'Master': 5,
                    'Professional': 6,
                    'Train': 2.5
                }
                
                # Convert Qualification to numerical ranges
                df['Qualification'] = df['Qualification'].map(ranges)
                ''', language='python'
        )
        st.write('##### Job Type')
        st.code(
                '''
                # Define the ranges for JobType
                ranges = {
                    'Full Time': 3,
                    'Temporary Contract': 2,
                    'Part Time': 1
                }
                
                # Convert JobType to numerical ranges
                df['JobType'] = df['JobType'].map(ranges)
                ''', language='python'
        )
        st.write('##### Position Level')
        st.code(
                '''
                # Define the ranges for PositionLevel
                ranges = {
                    'Kindergarten': 1,
                    'Fresh': 2,
                    'Junior Executive': 4,
                    'Non-Executive': 3,
                    'Senior Executive': 5,
                    'Manager': 6,
                    'Senior Manager': 7
                }
                
                # Convert PositionLevel to numerical ranges
                df['PositionLevel'] = df['PositionLevel'].map(ranges)
                ''', language='python'
        )
        st.write('##### Location')
        st.code(
                '''
                # Define the ranges for Location
                ranges = {
                    'Cambodia': 1,
                    'Phnom Penh': 20,
                    'Preah Sihanouk': 19,
                    'Battambang': 4,
                    'Siem Reap': 5,
                    'Bavet': 6,
                    'Poipet': 7,
                    'Kampong Speu': 8,
                    'Kampot': 9,
                    'Kandal': 10,
                    'Kampong Cham': 11,
                    'Svay Rieng': 18,
                    'Mondulkiri': 13,
                    'Banteay Meanchey': 14,
                    'Preah Vihear': 15,
                    'Pursat': 16,
                    'Kampong Chhnang': 17,
                    'Koh Kong': 18,
                    'Kampong Thom': 19,
                    'Tbong Khmum': 21
                }
                
                # Convert Location to numerical ranges
                df['Location'] = df['Location'].map(ranges)
                ''', language='python'
        )
        st.write('##### Salary')
        st.code(
                '''
                ranges = {
                    (200, 500): 0,
                    (500, 1000): 1,
                    (1000, 1500): 2,
                    (1500, np.inf): 3
                }
                
                # Convert Salary_min to numerical representations
                df['Salary_min'] = pd.cut(df['Salary_min'], bins=[200, 500, 1000, 1500, np.inf], labels=[0, 1, 2, 3])
                
                # Convert the datatype to int
                df['Salary_min'] = df['Salary_min'].astype(int)
                ''', language='python'
        )
feature_engineering()

# st.sidebar.write("[7. Model Building](#model-building)")
# st.sidebar.write("[7.1. Train Split Test](#train_split)")
# st.sidebar.write("[7.2. Classification](#classification)")
# st.sidebar.write("[7.2.1. K Nearest Neighbor (KNN)](#knn)")
def pre_qual():
    # Define the ranges for qualification
    ranges = {
        'No': 0,
        'Others': 0,
        'Diploma': 1,
        'Associate': 3,
        'Bacc2': 2,
        'Bachelor': 4,
        'Master': 5,
        'Professional': 6,
        'Train': 2.5
    }
    # Convert Qualification to numerical ranges
    df['Qualification'] = df['Qualification'].map(ranges)

def pre_jobtype():
    # Define the ranges for JobType
    ranges = {
        'Full Time': 3,
        'Temporary Contract': 2,
        'Part Time': 1
    }
    # Convert JobType to numerical ranges
    df['JobType'] = df['JobType'].map(ranges)

def pre_poslvl():
    # Define the ranges for PositionLevel
    ranges = {
        'Kindergarten': 1,
        'Fresh': 2,
        'Junior Executive': 4,
        'Non-Executive': 3,
        'Senior Executive': 5,
        'Manager': 6,
        'Senior Manager': 7
    }

    # Convert PositionLevel to numerical ranges
    df['PositionLevel'] = df['PositionLevel'].map(ranges)

def pre_loc():
    # Define the ranges for Location
    ranges = {
        'Cambodia': 1,
        'Phnom Penh': 20,
        'Preah Sihanouk': 19,
        'Battambang': 4,
        'Siem Reap': 5,
        'Bavet': 6,
        'Poipet': 7,
        'Kampong Speu': 8,
        'Kampot': 9,
        'Kandal': 10,
        'Kampong Cham': 11,
        'Svay Rieng': 18,
        'Mondulkiri': 13,
        'Banteay Meanchey': 14,
        'Preah Vihear': 15,
        'Pursat': 16,
        'Kampong Chhnang': 17,
        'Koh Kong': 18,
        'Kampong Thom': 19,
        'Tbong Khmum': 21
    }
    # Convert Location to numerical ranges
    df['Location'] = df['Location'].map(ranges)

def pre_salary():
    ranges = {
        (200, 500): 0,
        (500, 1000): 1,
        (1000, 1500): 2,
        (1500, np.inf): 3
    }
    # Convert Salary_min to numerical representations
    df['Salary_min'] = pd.cut(df['Salary_min'], bins=[200, 500, 1000, 1500, np.inf], labels=[0, 1, 2, 3])

    # Convert the datatype to int
    df['Salary_min'] = df['Salary_min'].astype(int)

def train_split():
    y = df['Salary_min']
    X = df.drop(columns='Salary_min', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
    return f'Train set: {X_train.shape}, {y_train.shape} \n' \
           f'Test set: {X_test.shape}, {y_test.shape}'

st.markdown("""<div id="model-building"></div>""", unsafe_allow_html=True)
st.header("7. Model Building")
with st.expander("CLICK HERE TO SHOW DATA"):
    st.markdown("""<div id="train_split"></div>""", unsafe_allow_html=True)
    st.subheader('7.1. Train Split Test')
    # with st.expander("CLICK HERE TO SHOW DATA"):
    st.code(
        '''
        from sklearn.model_selection import train_test_split
        y = df.Salary_min
        X = df.drop(columns='Salary_min', axis = 1)
        X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
        print('Train set:', X_train.shape,  y_train.shape)
        print('Test set:', X_test.shape,  y_test.shape)
        ''', language='python'
    )
    s = train_split()
    st.text(s)
    st.markdown("""<div id="classification"></div>""", unsafe_allow_html=True)
    st.subheader('7.2. Classification')
    # with st.expander("CLICK HERE TO SHOW DATA"):
    st.write('''
    #### 7.2.1. K Nearest Neighbor (KNN)

    ##### What about other K?

    K in KNN, is the number of nearest neighbors to examine. It is supposed to be specified by the user. 
    So, how can we choose right value for K? The general solution is to reserve a part of your data for testing the 
    accuracy of the model. Then choose k =1, use the training part for modeling, and calculate the accuracy of prediction 
    using all samples in your test set. Repeat this process, increasing the k, and see which k is the best for your model.
    We can calculate the accuracy of KNN for different values of k.

    Classifier implementing the k-nearest neighbors vote.
    ''')
    st.code(
        '''
        df.head()
        ''', language='python'
    )
    pre_qual()
    pre_jobtype()
    pre_poslvl()
    pre_loc()
    pre_salary()
    st.table(df.head())
    st.code(
        '''
        from sklearn import preprocessing
        X_train_norm = preprocessing.StandardScaler().fit(X_train).transform(X_train.astype(float))
        X_train_norm[0:5]
        ''', language='python'
    )
    y = df['Salary_min']
    X = df.drop(columns='Salary_min', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
    X_train_norm = preprocessing.StandardScaler().fit(X_train).transform(X_train.astype(float))
    st.table(X_train_norm[0:5])
    st.code(
        '''
        from sklearn.neighbors import KNeighborsClassifier
        # Train Model and Predict 
        neigh = KNeighborsClassifier(n_neighbors = 7).fit(X_train_norm,y_train)
        neigh
        ''', language='python'
        )
    # Train Model and Predict
    neigh = KNeighborsClassifier(n_neighbors=7).fit(X_train_norm, y_train)
    st.write(neigh)
    st.write(
        '''
        ##### Predicting
        We can use the model to make predictions on the test set:
        '''
    )
    st.code(
        '''
        X_test_norm = preprocessing.StandardScaler().fit(X_test).transform(X_test.astype(float))
        X_test_norm[0:5]    
        ''', language='python'
    )
    X_test_norm = preprocessing.StandardScaler().fit(X_test).transform(X_test.astype(float))
    st.table(X_test_norm[0:5])
    st.code(
        '''
        yhat = neigh.predict(X_test_norm)
        yhat[0:10]
        ''', language='python'
    )
    yhat = neigh.predict(X_test_norm)
    st.dataframe(yhat[0:10])

    st.markdown("""<div id="accuracy-evaluation"></div>""", unsafe_allow_html=True)
    st.subheader('7.3. Accuracy Evaluation')
    # with st.expander("CLICK HERE TO SHOW DATA"):
    st.write(
        '''
        In multilabel classification, accuracy classification score is a function that computes subset accuracy. 
        This function is equal to the jaccard_score function. Essentially, it calculates how closely the actual 
        labels and predicted labels are matched in the test set.
        '''
    )
    st.code(
        '''
        from sklearn import metrics
        print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train_norm)))
        print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))
        ''', language='python'
    )
    st.text(f"Train set Accuracy: {metrics.accuracy_score(y_train, neigh.predict(X_train_norm))}")
    st.text(f"Test set Accuracy: {metrics.accuracy_score(y_test, yhat)}")

# Fature Selection
y = df['Salary_min']
X = df.drop(columns='Salary_min', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
X_train_norm = preprocessing.StandardScaler().fit(X_train).transform(X_train.astype(float))
# Create a K-Nearest Neighbors classifier
knn = KNeighborsClassifier(n_neighbors=7)
# Perform feature selection with SelectKBest using chi-square
# num_features_to_select= int(input('Input the number of Feature: '))
num_features_to_select = 4
selector = SelectKBest(score_func=chi2, k=num_features_to_select)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)
# Get the selected feature indices
selected_feature_indices = selector.get_support(indices=True)
# Get the names of the selected features
selected_feature_names = df.drop(columns='Salary_min', axis=1).columns[selected_feature_indices]
# Train and test the KNN classifier with the selected features
knn.fit(X_train_selected, y_train)
accuracy = knn.score(X_test_selected, y_test)

print(f"Selected features: {selected_feature_names}")
print(f"Feature scores: {selector.scores_[selected_feature_indices]}")
print(f"Accuracy with {num_features_to_select} features: {accuracy:.2f}")

# Feature Selection
st.markdown("""<div id="feat-selection"></div>""", unsafe_allow_html=True)
st.header('8. Feature Selection')
with st.expander("CLICK HERE TO SHOW DATA"):
    st.code(
        '''
        from sklearn.feature_selection import SelectKBest, chi2
        from sklearn.feature_selection import RFE
        # Create a K-Nearest Neighbors classifier
        knn = KNeighborsClassifier(n_neighbors=5)

        # Perform feature selection with SelectKBest using chi-square

        num_features_to_select=4
        selector = SelectKBest(score_func=chi2, k=num_features_to_select)
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)

        # Get the selected feature indices
        selected_feature_indices = selector.get_support(indices=True)

        # Get the names of the selected features
        selected_feature_names = df.drop(columns='Salary_min', axis=1).columns[selected_feature_indices]

        # Train and test the KNN classifier with the selected features
        knn.fit(X_train_selected, y_train)
        accuracy = knn.score(X_test_selected, y_test)

        print(f"Selected features: {selected_feature_names}")
        print(f"Feature scores: {selector.scores_[selected_feature_indices]}")
        print(f"Accuracy with {num_features_to_select} features: {accuracy:.2f}")
        ''', language='python'
    )

    st.text(f"Selected features: {selected_feature_names}")
    st.text(f"Feature scores: {selector.scores_[selected_feature_indices]}")
    st.text(f"Accuracy with {num_features_to_select} features: {accuracy:.2f}")

    st.code(
        '''
        df.shape
        ''', language='python'
    )
    st.text(df.shape)

def plot_confusion_matrix(cm, classes,
                             normalize=False,
                             title='Confusion matrix',
                             cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

st.markdown("""<div id="svm"></div>""", unsafe_allow_html=True)
st.header('9. SVM (Support Vector Machine) with Scikit-learn')
with st.expander("CLICK HERE TO SHOW DATA"):
    st.subheader('9.1. SVM with Polynomial Kernel')
    st.write(
        '''
        The SVM algorithm offers a choice of kernel functions for performing its processing. Basically, mapping data 
        into a higher dimensional space is called kernelling. The mathematical function used for the transformation is 
        known as the kernel function, and can be of different types, such as:

            1.Linear
            2.Polynomial
            3.Radial basis function (RBF)
            4.Sigmoid

        Each of these functions has its characteristics, its pros and cons, and its equation, but as there's no easy 
        way of knowing which function performs best with any given dataset. We usually choose different functions in 
        turn and compare the results. Let's just use the default, RBF (Radial Basis Function) for this lab.
        '''
    )
    y = df['Salary_min']
    X = df.drop(columns='Salary_min', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
    X_train_norm = preprocessing.StandardScaler().fit(X_train).transform(X_train.astype(float))
    st.code(
        '''
        from sklearn import svm
        clf = svm.SVC(kernel='poly')
        clf.fit(X_train, y_train) 
        ''', language='python'
    )

    from sklearn import svm
    clf = svm.SVC(kernel='poly')
    st.write(clf.fit(X_train, y_train))
    st.write('After being fitted, the model can then be used to predict new values:')
    st.code(
        '''
        yhat = clf.predict(X_test)
        yhat [0:5]
        ''', language='python'
    )
    yhat = clf.predict(X_test)
    st.write(yhat[0:5])
    st.subheader('9.2. Evaluation')
    st.code(
        '''
        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
        import itertools
        def plot_confusion_matrix(cm, classes,
                    normalize=False,
                    title='Confusion matrix',
                    cmap=plt.cm.Blues):
            """
            This function prints and plots the confusion matrix.
            Normalization can be applied by setting `normalize=True`.
            """
            if normalize:
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                print("Normalized confusion matrix")
            else:
                print('Confusion matrix, without normalization')
    
            print(cm)
    
            plt.imshow(cm, interpolation='nearest', cmap=cmap)
            plt.title(title)
            plt.colorbar()
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes, rotation=45)
            plt.yticks(tick_marks, classes)
    
            fmt = '.2f' if normalize else 'd'
            thresh = cm.max() / 2.
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(j, i, format(cm[i, j], fmt),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
    
            plt.tight_layout()
            plt.ylabel('True label')
            plt.xlabel('Predicted label')

        # Compute confusion matrix
        cnf_matrix = confusion_matrix(y_test, yhat, labels=[2,4])
        np.set_printoptions(precision=2)

        print(classification_report(y_test, yhat))

        # Plot non-normalized confusion matrix
        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=['Benign(2)','Malignant(4)'],normalize= False,  title='Confusion matrix')
        ''', language='python'
    )
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, yhat, labels=[2, 4])
    np.set_printoptions(precision=2)

    st.text(classification_report(y_test, yhat))
    st.text(f'Confusion matrix, without normalization\n{cnf_matrix}')

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix,
                          classes=['Benign(2)', 'Malignant(4)'],
                          normalize=False,
                          title='Confusion matrix')
    # confusion matrix image graph
    confusion()

    st.code(
        '''
        confusion_matrix(y_test, yhat)
        ''', language='python'
    )
    st.text(confusion_matrix(y_test, yhat))
    st.code(
        '''
        accuracy_score(y_test, yhat)
        ''', language='python'
    )
    st.text(accuracy_score(y_test, yhat))

st.markdown("""<div id="gbc"></div>""", unsafe_allow_html=True)
st.header('10. Gradient Boosting Classifier')
with st.expander("CLICK HERE TO SHOW DATA"):
    st.code(
        '''
        from sklearn.ensemble import GradientBoostingClassifier

        # Initialize the Gradient Boosting Classifier with default hyperparameters
        gb_classifier = GradientBoostingClassifier()
        
        # Train the model on the training data
        gb_classifier.fit(X_train_norm, y_train)
        # Assuming you have also prepared the test data X_test_norm
        y_pred = gb_classifier.predict(X_test_norm)
        ''', language='python'
    )
    # Initialize the Gradient Boosting Classifier with default hyperparameters
    gb_classifier = GradientBoostingClassifier()

    # Train the model on the training data
    gb_classifier.fit(X_train_norm, y_train)
    # Assuming you have also prepared the test data X_test_norm
    y_pred = gb_classifier.predict(X_test_norm)

    st.code(
        '''
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.2f}")
        
        # Generate classification report
        print(classification_report(y_test, y_pred))
        
        # Create and display confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:")
        print(conf_matrix)
        ''', language='python'
    )
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    st.text(f"Accuracy: {accuracy:.2f}")

    # Generate classification report
    st.text(classification_report(y_test, y_pred))

    # Create and display confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    st.text("Confusion Matrix:")
    st.text(conf_matrix)

    st.code(
        '''
        from sklearn.feature_selection import RFE
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.metrics import accuracy_score
        
        # Create a GradientBoostingClassifier
        gb_classifier = GradientBoostingClassifier()
        
        # Perform feature selection with RFE
        num_features_to_select = int(input('Input the number of Feature: '))
        
        selector = RFE(gb_classifier, n_features_to_select=num_features_to_select)
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)
        
        # Get the selected feature indices
        selected_feature_indices = selector.support_
        
        # Get the names of the selected features
        selected_feature_names = df.drop(columns='Salary_min', axis=1).columns[selected_feature_indices]
        
        # Train and test the GradientBoostingClassifier with the selected features
        gb_classifier.fit(X_train_selected, y_train)
        y_pred = gb_classifier.predict(X_test_selected)
        
        # Calculate accuracy with the selected features
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Selected features: {selected_feature_names}")
        print(f"Feature ranking: {selector.ranking_}")
        print(f"Accuracy with {num_features_to_select} features: {accuracy:.2f}")
        ''', language='python'
    )
    # Create a GradientBoostingClassifier
    gb_classifier = GradientBoostingClassifier()

    # Perform feature selection with RFE
    # num_features_to_select = int(input('Input the number of Feature: '))
    num_features_to_select = 4
    selector = RFE(gb_classifier, n_features_to_select=num_features_to_select)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)

    # Get the selected feature indices
    selected_feature_indices = selector.support_

    # Get the names of the selected features
    selected_feature_names = df.drop(columns='Salary_min', axis=1).columns[selected_feature_indices]

    # Train and test the GradientBoostingClassifier with the selected features
    gb_classifier.fit(X_train_selected, y_train)
    y_pred = gb_classifier.predict(X_test_selected)

    # Calculate accuracy with the selected features
    accuracy = accuracy_score(y_test, y_pred)

    st.text(f"Selected features: {selected_feature_names}")
    st.text(f"Feature ranking: {selector.ranking_}")
    st.text(f"Accuracy with {num_features_to_select} features: {accuracy:.2f}")

    st.code(
        '''
        accuracy_score(y_test, y_pred)
        ''', language='python'
    )
    st.text(f"Accuracy test: {accuracy_score(y_test, y_pred)}")

    st.code(
        '''
        # Perform feature selection with RFE
        num_features_to_select = int(input('Input the number of Feature: '))
        
        selector = RFE(gb_classifier, n_features_to_select=num_features_to_select)
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)
        
        # Train and test the GradientBoostingClassifier with the selected features
        gb_classifier.fit(X_train_selected, y_train)
        y_pred = gb_classifier.predict(X_test_selected)
        
        # Calculate accuracy with the selected features
        accuracy = accuracy_score(y_test, y_pred)
        
        # Get the selected feature names
        selected_feature_names = df.drop(columns='Salary_min', axis=1).columns[selector.support_]
        
        print(f"Selected features: {selected_feature_names}")
        print(f"Feature ranking: {selector.ranking_}")
        print(f"Accuracy with {num_features_to_select} features: {accuracy:.2f}")
        
        # Get feature importances after RFE
        feature_importance = pd.Series(gb_classifier.feature_importances_, index=selected_feature_names).sort_values(ascending=False)
        
        # Plot the feature importance
        plt.figure(figsize=(12, 8))
        plt.title("Feature Importance")
        ax = sns.barplot(y=feature_importance.index, x=feature_importance.values, palette='dark', orient='h')
        plt.show()
        ''', language='python'
    )

    # Perform feature selection with RFE
    # num_features_to_select = int(input('Input the number of Feature: '))

    num_features_to_select = 4
    selector = RFE(gb_classifier, n_features_to_select=num_features_to_select)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)

    # Train and test the GradientBoostingClassifier with the selected features
    gb_classifier.fit(X_train_selected, y_train)
    y_pred = gb_classifier.predict(X_test_selected)

    # Calculate accuracy with the selected features
    accuracy = accuracy_score(y_test, y_pred)

    # Get the selected feature names
    selected_feature_names = df.drop(columns='Salary_min', axis=1).columns[selector.support_]

    st.text(f"Selected features: {selected_feature_names}")
    st.text(f"Feature ranking: {selector.ranking_}")
    st.text(f"Accuracy with {num_features_to_select} features: {accuracy:.2f}")

    # Get feature importance after RFE
    feature_importance = pd.Series(gb_classifier.feature_importances_, index=selected_feature_names).sort_values(
        ascending=False)

    # Plot the feature importance
    plt.figure(figsize=(12, 8))
    plt.title("Feature Importance")
    ax = sns.barplot(y=feature_importance.index, x=feature_importance.values, palette='dark', orient='h')
    plt.show()

    # feature importance image function
    feat_importance()

# Result
st.markdown("""<div id="result"></div>""", unsafe_allow_html=True)
st.header('11. Result')
with st.expander("CLICK HERE TO SHOW DATA"):
    st.write("##### KNN Classification")
    st.text(
        '''
        Selected features: Index(['PositionLevel', 'Location', 'WorkingExperience', 'Qualification'], dtype='object')
        Feature scores: [11.93 19.07 59.19  7.85]
        Accuracy with 4 features: 0.49
        '''
    )

# Report
def show_pdf_report(pdf_file):
    with open(pdf_file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(base64.b64decode(base64_pdf))
    temp_file.close()

    st.markdown(get_pdf_download_link_report(temp_file.name), unsafe_allow_html=True)


def get_pdf_download_link_report(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")

    download_link_report = f'<a href="data:application/pdf;base64,{base64_pdf}" download="report_file.pdf">Click here to download the PDF file</a>'
    return download_link_report


# Guideline
def show_pdf_guideline(pdf_file):
    with open(pdf_file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(base64.b64decode(base64_pdf))
    temp_file.close()

    st.markdown(get_pdf_download_link_guideline(temp_file.name), unsafe_allow_html=True)

# Download Report file
def get_pdf_download_link_guideline(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")

    download_link_guideline = f'<a href="data:application/pdf;base64,{base64_pdf}" download="guideline_file.pdf">Click here to download the PDF file</a>'
    return download_link_guideline


st.markdown("""<div id="report"></div>""", unsafe_allow_html=True)
st.markdown("### Job Analysis Report")
st.markdown("👇 Job Analysis Report | Click the link down below to download 📑")
show_pdf_report("PDF Report/Report.pdf")

st.markdown("""<div id="guideline"></div>""", unsafe_allow_html=True)
st.markdown("### Job Analysis Guideline")
st.markdown("👇 Job Analysis Guideline | Click the link down below to download 📑")
show_pdf_guideline("PDF Report/Guideline.pdf")
