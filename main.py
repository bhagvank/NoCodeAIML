import streamlit as st
import pandas as pd
import numpy as np

from sidebars.anomaly_detection_sidebars import kNN_ad_sidebar
from sidebars.anomaly_detection_sidebars import LOF_sidebar
from sidebars.anomaly_detection_sidebars import iForest_sidebar
from sidebars.classification_sidebars import kNN_sidebar
from sidebars.classification_sidebars import SVM_sidebar
from sidebars.classification_sidebars import Logistic_Regression_sidebar
from sidebars.classification_sidebars import RF_sidebar
from sidebars.classification_sidebars import Decision_Trees_sidebar
from sidebars.clustering_sidebars import DBSCAN_sidebar
from sidebars.clustering_sidebars import KMEANS_sidebar
from sidebars.clustering_sidebars import OPTICS_sidebar
import base64
from jinja2 import Environment, FileSystemLoader
import os
import collections
import utils
from videoprocessing.data import *
from videoprocessing.input import image_input, webcam_input
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import altair as alt
from bokeh.plotting import figure
from make_plots import (
    matplotlib_plot,
    sns_plot,
    pd_plot,
    plotly_plot,
    altair_plot,
    bokeh_plot,
)

from draw import (
      about,
      full_app,
      center_circle_app,
      color_annotation_app,
      png_export,
      compute_arc_length,
)

import spacy_streamlit
from pathlib import Path
import srsly
import importlib
import random

from io import BytesIO
from pathlib import Path
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from svgpathtools import parse_path

import SessionState

st.set_page_config(layout="wide")

def main():
 st.title('No Code AI/ML Platform')


 st.sidebar.title("No Code AI/ML Platform")
 app_mode = st.sidebar.selectbox("Choose the option",
        ["Data Analysis", "Run the Algorithm","Process Images", "Draw a Plot","Video Analysis","NLP","Drawing"])
 if app_mode == "Data Analysis":
        data_analysis()
        #st.sidebar.success('To continue select "Run the app".')
 elif app_mode == "Run the Algorithm":
        run_algorithm()
 elif app_mode == "Process Images":
       process_images()
 elif app_mode == "Draw a Plot":
       draw_a_plot()
 elif app_mode == "Video Analysis":
       analyse_video()
 elif app_mode == "NLP":
       nlp_show() 
 elif app_mode == "Drawing":
       draw()              


def data_analysis():

 st.title('Data Analysis')

 uploaded_file = st.file_uploader("Choose a file")


 if uploaded_file is not None:
    @st.cache
    def load_data(nrows):

      data = pd.read_csv(uploaded_file, nrows=nrows)
      lowercase = lambda x: str(x).lower()
      data.rename(lowercase, axis='columns', inplace=True)
      #data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
      return data

    data_load_state = st.text('Loading data...')
    data = load_data(10000)
    data_load_state.text("Done! (using st.cache)")

    if st.checkbox('Show raw data'):
       st.subheader('Raw data')
       st.write(data)
    #return data

def run_algorithm():



     templates = {
    'Anomaly Detection': {
        'LOF': 'templates/Anomaly Detection/LOF',
        'iForest': 'templates/Anomaly Detection/iForest',
        'kNN': 'templates/Anomaly Detection/kNN'
    },
    'Classification': {
        'Logistic Regression': 'templates/Classification/Logistic Regression',
        'kNN': 'templates/Classification/kNN',
        'SVM': 'templates/Classification/SVM',
        'Random Forest': 'templates/Classification/Random Forest',
        'Decision Tree': 'templates/Classification/Decision Trees'
    },
    'Clustering': {
        'DBSCAN': 'templates/Clustering/DBSCAN',
        'K-Means': 'templates/Clustering/K-Means',
        'OPTICS': 'templates/Clustering/OPTICS',
    }
}

     with st.sidebar:
      st.write("## Choose Task")
      task = st.selectbox("Task", list(templates.keys()))




      if isinstance(templates[task], dict):
         algorithm = st.sidebar.selectbox(
            "Which Algorithm?", list(templates[task].keys())
         )
         template_path = templates[task][algorithm]
      else:
         template_path = templates[task]
      if task == "Anomaly Detection":
         if algorithm == 'LOF':
             inputs = LOF_sidebar()
         if algorithm == "iForest":
            inputs = iForest_sidebar()
         if algorithm == "kNN":
             inputs = kNN_ad_sidebar()
      if task == "Classification":
         if algorithm == "Logistic Regression":
             inputs = Logistic_Regression_sidebar()
         if algorithm == 'kNN':
             inputs = kNN_sidebar()
         if algorithm == 'SVM':
             inputs = SVM_sidebar()
         if algorithm == "Random Forest":
             inputs = RF_sidebar()
         if algorithm == "Decision Tree":
             inputs = Decision_Trees_sidebar()
     if task == "Clustering":
         if algorithm == "DBSCAN":
             inputs = DBSCAN_sidebar()
         if algorithm == "K-Means":
             inputs = KMEANS_sidebar()
         if algorithm == "OPTICS":
             inputs = OPTICS_sidebar()

     env = Environment(loader=FileSystemLoader(template_path), trim_blocks=True, lstrip_blocks=True)

     template = env.get_template("code-template.py.jinja")
     code = template.render(header=header, **inputs)

     st.title("Algorithm  For "+ task)

     if st.checkbox('Execute the Algorithm :'+algorithm):

       with st.echo():

          os.system("python3 -c '"+code +"' > output.txt")

       show_results()




def show_results():

     st.title("Showing Results")

     data = pd.read_csv("output.txt")

     if st.checkbox('Show Output'):
       st.subheader('Output')
       st.write(data)


def draw():
    
    st.title("Drawable Canvas")
    st.sidebar.subheader("Configuration")
    session_state = SessionState.get(button_id="", color_to_label={})
    PAGES = {
        "About": about,
        "Basic example": full_app,
        "Get center coords of circles": center_circle_app,
        "Color-based image annotation": color_annotation_app,
        "Download Base64 encoded PNG": png_export,
        "Compute the length of drawn arcs": compute_arc_length,
    }
    page = st.sidebar.selectbox("Page:", options=list(PAGES.keys()))
    PAGES[page](session_state)

    #with st.sidebar:
    #    st.markdown("---")
    #    st.markdown(
    #        '<h6>Made in &nbsp<img src="https://streamlit.io/images/brand/streamlit-mark-color.png" alt="Streamlit logo" height="16">&nbsp by <a href="https://twitter.com/andfanilo">@andfanilo</a></h6>',
    #        unsafe_allow_html=True,
    #    )
    #    st.markdown(
    #        '<div style="margin: 0.75em 0;"><a href="https://www.buymeacoffee.com/andfanilo" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/default-orange.png" alt="Buy Me A Coffee" height="41" width="174"></a></div>',
    #        unsafe_allow_html=True,
    #    )
    

def process_images():

    template_dict = collections.defaultdict(dict)
    template_dirs = [
       f for f in os.scandir("imageprocessing/templates") if f.is_dir() and f.name != "example"
       ]

    template_dirs = sorted(template_dirs, key=lambda e: e.name)
    for template_dir in template_dirs:
      try:
         # Templates with task + framework.
         task, framework = template_dir.name.split("_")
         template_dict[task][framework] = template_dir.path
      except ValueError:
         # Templates with task only.
         template_dict[template_dir.name] = template_dir.path

    with st.sidebar:
     #st.info(
     #   "🎈 **NEW:** Add your own code template to this site! [Guide](https://github.com/jrieke/traingenerator#adding-new-templates)"
   # )
    # st.error(
    #     "Found a bug? [Report it](https://github.com/jrieke/traingenerator/issues) 🐛"
    # )
     st.write("## Task")
     task = st.selectbox(
        "Which problem do you want to solve?", list(template_dict.keys())
    )
     if isinstance(template_dict[task], dict):
        framework = st.selectbox(
            "In which framework?", list(template_dict[task].keys())
        )
        template_dir = template_dict[task][framework]
     else:
        template_dir = template_dict[task]


# Show template-specific sidebar components (based on sidebar.py in the template dir).
    template_sidebar = utils.import_from_file(
      "template_sidebar", os.path.join(template_dir, "sidebar.py")
       )
    inputs = template_sidebar.show()


# Generate code and notebook based on template.py.jinja file in the template dir.
    env = Environment(
     loader=FileSystemLoader(template_dir), trim_blocks=True, lstrip_blocks=True,
 )
    template = env.get_template("code-template.py.jinja")
    code = template.render(header=utils.code_header, notebook=False, **inputs)
 
    #st.code(code)

    st.title("Algorithm  For "+ task)

    if st.checkbox('Execute the Algorithm :'+task):

       with st.echo():

        os.system("python3 -c '"+code +"' > output.txt")

       show_results()

def draw_a_plot():
    
    

    plot_types = (
        "Scatter",
        "Histogram",
        "Bar",
        "Line",
        "3D Scatter",
    )  # maybe add 'Boxplot' after fixes
    libs = (
        "Matplotlib",
        "Seaborn",
        "Plotly Express",
        "Altair",
        "Pandas Matplotlib",
        "Bokeh",
    )
    datasets = (  "penguins",)
                 
                 
                # "anagrams",
                # "anscombe",
                # "attention",
                # "brain_networks",
                # "car_crashes",
                # "diamonds",
                # "dots",
                # "exercise",
                # "flights",
                # "fmri",
                # "gammas",
                # "geyser",
                # "iris",
                # "mpg",
                
                # "planets",
                # "tips",
                # "titanic",
    
               #)
    
    
    
    
    
    #with st.beta_container():
        #st.title("Python Data Visualization Tour")
        #st.header("Popular plots in popular plotting libraries")
        #st.write("""See the code and plots for five libraries at once.""")


    # User choose user type
    with st.sidebar:
         chart_type = st.selectbox("Choose your chart type", plot_types)
         kind = st.selectbox("Choose your Plot Library", libs)
         dataset = st.selectbox("Choose your Dataset", datasets)
         
    with st.beta_container():
        st.subheader(f"Plotting:  {chart_type}")
        st.write("")
    
    #df = data_analysis()
    pens_df = load_data(dataset=dataset)
    df = pens_df.copy()
    df.index = pd.date_range(start="1/1/18", periods=len(df), freq="D")
    #st.write(pens_df.head())
    with st.sidebar:
         category = st.selectbox("Choose your Category", list(df.columns.values))
         chart_x =  st.selectbox("Choose your X Axis",  list(df.columns.values))
         chart_y =  st.selectbox("Choose your Y Axis",  list(df.columns.values))
         chart_z = st.selectbox("Choose your Z Axis",  list(df.columns.values))
    
    category_list = list(df[category].unique())
    
    #st.write(category_list)
    with st.beta_container():
        show_data = st.checkbox("See the raw data?")

        if show_data:
            df

    #two_cols = st.checkbox("2 columns?", True)
    #if two_cols:
    #    col1, col2 = st.beta_columns(2)
        
    # output plots
    #if two_cols:
    #    with col1:
    #        show_plot(kind="Matplotlib",chart_type=chart_type,df=df)
    #    with col2:
    #        show_plot(kind="Seaborn",chart_type=chart_type,df=df)
    #    with col1:
    #        show_plot(kind="Plotly Express",chart_type=chart_type,df=df)
    #    with col2:
    #        show_plot(kind="Altair",chart_type=chart_type,df=df)
    #    with col1:
    #        show_plot(kind="Pandas Matplotlib",chart_type=chart_type,df=df)
    #    with col2:
    #        show_plot(kind="Bokeh",chart_type=chart_type,df=df)
    #else:
    with st.beta_container():
            #for lib in libs:
            show_plot(kind=kind,chart_type=chart_type,df=df)
    


        # notes
        #st.subheader("Notes")
        #st.write(
        #    """
        #    - This app uses [Streamlit](https://streamlit.io/) and the [Palmer #Penguins](https://allisonhorst.github.io/palmerpenguins/) dataset.      
#            - To see the full code check out the [GitHub repo](https://github.com/discdiver/data-viz-streamlit).
#            - Plots are interactive where that's the default or easy to add.
#            - Plots that use MatPlotlib under the hood have fig and ax objects defined before the code shown.
#            - Lineplots should have sequence data, so I created a date index with a sequence of dates for them. 
#            - Where an axis label shows by default, I left it at is. Generally where it was missing, I added it.
#            - There are multiple ways to make some of these plots.
#            - You can choose to see two columns, but with a narrow screen this will switch to one column automatically.
#            - Python has many data visualization libraries. This gallery is not exhaustive. If you would like to add #code for another library, please submit a [pull request](https://github.com/discdiver/data-viz-streamlit).
 #           - For a larger tour of more plots, check out the [Python Graph #Gallery](https://www.python-graph-gallery.com/density-plot/) and [Python Plotting for Exploratory Data #Analysis](https://pythonplot.com/).
 #           - The interactive Plotly Express 3D Scatterplot is cool to play with. Check it out! 😎
  #      
   #         Made by [Jeff Hale](https://www.linkedin.com/in/-jeffhale/). 
        
    #        Subscribe to my [Data Awesome newsletter](https://dataawesome.com) for the latest tools, tips, and resources.
     #       """
      #  )                
 
 
def analyse_video():
    st.title("Neural Style Transfer")
    st.sidebar.title('Navigation')
    method = st.sidebar.radio('Go To ->', options=['Webcam', 'Image'])
    st.sidebar.header('Options')

    style_model_name = st.sidebar.selectbox("Choose the style model: ", style_models_name)

    if method == 'Image':
        image_input(style_model_name)
    else:
        webcam_input(style_model_name)   
        
        
def nlp_show():
    MODELS = srsly.read_json(Path(__file__).parent / "data/models.json")
    DEFAULT_MODEL = "en_core_web_sm"
    DEFAULT_TEXT = "David Bowie moved to the US in 1974, initially staying in New York City before settling in Los Angeles."
    DESCRIPTION = """**Explore trained [spaCy v3.0](https://nightly.spacy.io) pipelines**"""

    def get_default_text(nlp):
        try:
            examples = importlib.import_module(f".lang.{nlp.lang}.examples", "spacy")
            return examples.sentences[0]
        except (ModuleNotFoundError, ImportError):
            return ""

    spacy_streamlit.visualize(
        MODELS,
        default_model=DEFAULT_MODEL,
        visualizers=["parser", "ner", "similarity", "tokens"],
        show_visualizer_select=True,
        sidebar_description=DESCRIPTION,
        get_default_text=get_default_text
    )         
 
def header(text):
    l = int((70 - len(text))/2)
    return "#" + '='*(l-1) + " " + text + " " + '='*l



# get data
@st.cache(allow_output_mutation=True)
def load_data(dataset):
    return sns.load_dataset(dataset)

def show_plot(kind: str,chart_type,df):
    st.write(kind)
    if kind == "Matplotlib":
        plot = matplotlib_plot(chart_type, df)
        st.pyplot(plot)
    elif kind == "Seaborn":
        plot = sns_plot(chart_type, df)
        st.pyplot(plot)
    elif kind == "Plotly Express":
        plot = plotly_plot(chart_type, df)
        st.plotly_chart(plot, use_container_width=True)
    elif kind == "Altair":
        plot = altair_plot(chart_type, df)
        st.altair_chart(plot, use_container_width=True)
    elif kind == "Pandas Matplotlib":
        plot = pd_plot(chart_type, df)
        st.pyplot(plot)
    elif kind == "Bokeh":
        plot = bokeh_plot(chart_type, df)
        st.bokeh_chart(plot, use_container_width=True)

if __name__ == "__main__":
    #st.set_page_config(
    #    page_title="Streamlit Drawable Canvas", page_icon=":pencil2:"
    #)
    main()
