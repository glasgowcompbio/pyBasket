import seaborn as sns
import streamlit as st
import webbrowser
import altair as alt
import numpy as np
import pandas as pd
import dc_stat_think as dcst
import gzip
import pickle
from loguru import logger

if "data" in st.session_state:
    data = st.session_state["data"]

#Reads pickle file with results from pyBasket model
def readPickle(pick_file):
    try:
        with gzip.GzipFile(pick_file, 'rb') as f:
            return pickle.load(f)
    except OSError:
        logger.warning('Old, invalid or missing pickle in %s. '
                       'Please regenerate this file.' % pick_file)
        raise

#Sets the number of colours for a plot's palette depending on number of groups
def colours(num):
    if num > 2:
        palette = sns.color_palette("colorblind", num).as_hex()
    else:
        palette = ['#e50000','#02c14d']
    return palette

#Adds pyBasket logo in the sidebar
def add_logo():
    st.markdown(
        """
        <style>
            [data-testid="stSidebarNav"]::before {
                content: "EMERY";
                margin-left: 20px;
                margin-top: 20px;
                font-size: 30px;
                position: relative;
                top: 50px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

#Hides rows index in tables displayed
def hideRows():
    hide_table_row_index = """
                <style>
                thead tr th:first-child {display:none}
                tbody th {display:none}
                </style>
                """
    return hide_table_row_index

#Opens GeneCard page of the gene chosen
def openGeneCard(gene):
    webpage_link = "https://www.genecards.org/cgi-bin/carddisp.pl?gene=" + gene
    webbrowser.open(webpage_link)

#Saves non-interactive plots as a PNG in working directory
def savePlot(fig, feature):
    if st.button('Save Plot', key="plot"+feature):
        fig.save('plot_' + feature + '.png')
        st.info('Plot saved as .png in working directory', icon="ℹ️")
    else:
        st.write("")

#Saves interactive plots as a PNG in working directory
def savePlot_plt(fig,feature):
    if st.button('Save Plot', key="plot"+feature):
        fig.savefig(feature)
        st.info('Plot saved as .png in working directory', icon="ℹ️")
    else:
        st.write("")

#Saves table as a CSV in working directory
def saveTable(df, feature):
    if st.button('Save table', key="table_"+feature):
        df.to_csv('raw_data_'+feature+'.csv', index=False)
        st.info('Data saved as .csv file in working directory', icon="ℹ️")
    else:
        st.write("")

#Option to select basket and cluster interaction in the sidebar. If number of samples is more than 0, an option to save the samples in a
#CSV file will appear.
def sideBar():
    if "data" in st.session_state:
        data = st.session_state["data"]
        analysis_data = st.session_state["Analysis"]
        st.sidebar.title("Select basket*cluster interaction")
        with st.sidebar:
            cluster = st.selectbox("Select a cluster", data.clusters_names, key="cluster")
            if "cluster" not in st.session_state:
                st.session_state["cluster"] = cluster
            basket = st.selectbox("Select a basket", data.baskets_names, key="basket")
            if "basket" not in st.session_state:
                st.session_state["basket"] = basket
            subgroup, size = analysis_data.findInteraction(cluster, basket)
            st.info("###### Samples in **cluster {}** & **{} basket**: {}".format(cluster, basket, size))
            if size>0:
                if st.button('Save samples', key="table_samples"):
                    subgroup.to_csv('samples_interaction.csv', index=False)
                    st.info('Samples in interaction saved as .csv file in working directory', icon="ℹ️")
    else:
        with st.sidebar:
            st.warning("Results not found. Please upload in a results file in Home.")

#Interactive line graph
def alt_line_chart(df, df2, x, y, title_x, title_y, main_title,save):
    base = alt.Chart(df).mark_line().encode(
        alt.X(x+':Q', title=title_x),
        alt.Y(y+':Q', title=title_y)
    ).properties(
        height=650, width = 600,title=main_title
    )
    percentiles =alt.Chart(df2).mark_point(filled=True, size=300).encode(
        alt.X('x' + ':Q', title=title_x),
        alt.Y('y' + ':Q', title=title_y),color=alt.value('red')
        )
    labels = percentiles.mark_text(
        align='left',
        baseline='middle',
        dx=25,
        fontSize=15
    ).encode(
        text='y'
    )
    base = base + percentiles + labels
    savePlot(base, save)
    st.altair_chart(base, theme="streamlit", use_container_width=True)

#Interactive horizontal bar chart
def alt_hor_barplot(df, x, y, num_cols, title_x, title_y, colors,main_title,save):
    palette = colours(num_cols)
    base = alt.Chart(df).mark_bar().encode(
        alt.X(x, title=title_x),
        alt.Y(y+':N', title=title_y).sort('-x'), alt.Color(colors +':N')
    ).properties(
        height=650, width = 600,title=main_title
    ).configure_range(
        category=alt.RangeScheme(palette))
    savePlot(base,save)
    st.altair_chart(base, theme="streamlit", use_container_width=True)

#Interactive vertical bar chart
def alt_ver_barplot(df, x, y, num_cols, title_x, title_y, colors,main_title,save, tooltip):
    palette = colours(num_cols)
    base = alt.Chart(df).mark_bar(size=40).encode(
        alt.X(x +':N', title=title_x),
        alt.Y(y+':Q', title=title_y,axis=alt.Axis(grid=False)), alt.Color(colors+':N'), tooltip = tooltip
    ).properties(
        height=500, width = 600,title=main_title
    ).configure_range(
        category=alt.RangeScheme(palette))
    savePlot(base, save)
    st.altair_chart(base, theme="streamlit", use_container_width=True)

#Interactive scatterplot
def alt_scatterplot(df, x, y, title_x, title_y,main_title,save,tooltip ):
    base = alt.Chart(df).mark_circle(size=100).encode(
        x=alt.Y(x, title=title_x),
        y=alt.Y(y+':Q', title=title_y),
        tooltip=tooltip
    ).interactive().properties(height=400, width = 650,title=main_title)
    savePlot(base, save)
    st.altair_chart(base, theme="streamlit", use_container_width=True)

#Interactive boxplot
def alt_boxplot(df, x, y, num_cols, title_x, title_y, colors,main_title,save):
    palette = colours(num_cols)
    base = alt.Chart(df, title="AAC response").mark_boxplot(extent='min-max', ticks=True, size=50).encode(
        x=alt.X(x, title=title_x),
        y=alt.Y(y, title=title_y), color=alt.Color(colors + ':N')
    ).properties(height=650, width = 600, title = main_title).configure_range(category=alt.RangeScheme(palette))
    savePlot(base, save)
    st.altair_chart(base, theme="streamlit", use_container_width=True)

#Shows list of available transcripts and adds button with direct link to the selected gene's GeneCard page
def searchTranscripts(transcripts):
    transcript = st.selectbox("Select feature/transcript", transcripts, key="transcript")
    st.caption("Transcripts ordered by decreasing significance.")
    st.write(" ")
    st.write("Click button to search for feature {} in GeneCards database.".format(transcript))
    st.button('Open GeneCards', on_click=openGeneCard, args=(transcript,))
    st.write(" ")
    return transcript

#Applies ECDF and returns a table with results (Data exploration-pyBasket results page)
def ecdf(data,intervals):
    pct_val = np.percentile(data, intervals)
    x, y = dcst.ecdf(data)
    pct = pd.DataFrame({'Probability': x, 'Percent': y * 100})
    pct_val = pd.DataFrame({'x': np.round(pct_val,3), 'y': intervals})
    return pct, pct_val

#Applies ECDF for samples in a chosen basket-cluster interaction (Basket-Cluster interaction-Selected interaction page)
def ecdf_interaction(data,basket,cluster,RawD, cred_inter):
    intervals = np.array([100-cred_inter-5, 50, cred_inter+5])
    basket_index = data.baskets_names.index(basket)
    cluster_index = data.clusters_names.index(cluster)
    inferred_prob = data.stacked_posterior.joint_p[basket_index][cluster_index]
    pct, pct_val = ecdf(inferred_prob, intervals)
    title = "ECDF for "+ basket + "*" + str(cluster)+ " interaction"
    st.write("""##### {}th, 50th and {}th Percentile values are""".format(intervals[0], intervals[
        2]) + ": {0:.2f}, {1:.2f} and {2:.2f}".format(*pct_val['x']))
    if RawD:
        saveTable(pct, "raw-ecdf")
        st.dataframe(pct, use_container_width=True)
    else:
        alt_line_chart(pct,pct_val,'Probability', 'Percent', 'Probability', 'Percent', "Cumulative distribution for "+title,"ecdf")
        st.caption("The x-axis represents the probabilities points. The y-axis represents the proportion or fraction of data points that are less than or equal to a given value.")

#Function to find the set of samples based on a specified subgroup
def findSubgroup(option,feature):
    transcripts= data.expr_df_selected
    sub_patients = data.patient_df[data.patient_df[feature] == option]
    indexes = list(sub_patients.index.values)
    sub_transcript = transcripts.loc[indexes]
    return sub_transcript

