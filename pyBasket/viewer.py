import pickle
import tempfile

import streamlit as st

from clustering import plot_PCA
from common import load_obj
from interpret import get_predicted_basket_df, plot_basket_probs, get_basket_cluster_prob_df, \
    plot_basket_cluster_heatmap, find_top_k_indices, find_bottom_k_indices, get_coords, \
    select_partition, plot_responsive_count, get_member_expression, df_diff, ttest_dataframe, \
    plot_expression_boxplot


# Define the Streamlit app
def app():
    # Set the title and layout of the app
    st.set_page_config(page_title='pyBasket Visualisation', layout='wide')

    # Create the left sidebar with the file picker
    st.sidebar.title('File Picker')
    uploaded_file = st.sidebar.file_uploader('Upload your data file', type='p')

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(suffix=".py") as tmp_file:
            # dag_code = uploaded_file.getvalue().decode("utf-8")
            tmp_file.write(uploaded_file.getvalue())
            tmp_file.flush()
            save_data = load_obj(tmp_file.name)
            print(tmp_file.name)

            st.header('Basket trial visualisation')
            st.write('A basket trial is a type of clinical trial that tests the effectiveness of '
                     'a therapy in a group of patients who share a specific genetic alteration or '
                     'biomarker, regardless of their disease type or location in the body. '
                     'This approach is also referred to as a histology-agnostic trial, and it '
                     'is not limited to cancer. Basket trials can include patients with different '
                     'types of diseases, such as autoimmune disorders, rare genetic diseases, or '
                     'infectious diseases, who share a common molecular signature or biomarker. '
                     'This approach allows for a more personalized treatment approach and '
                     'potentially faster drug development for specific subpopulations of patients. '
                     'Basket trials can be more efficient and cost-effective than traditional '
                     'clinical trials, as they allow for multiple diseases to be studied '
                     'simultaneously, thus potentially reducing the number of patients '
                     'required for the study.')

            # plot predicted basket probabilities
            st.header('Response probability of baskets')
            st.write('The following plot shows the inferred response probability of each tissue'
                     ' (basket) based on the observed responses.')
            predicted_basket_df = get_predicted_basket_df(save_data)
            fig = plot_basket_probs(predicted_basket_df, return_fig=True)
            st.pyplot(fig)

            # create two columns for the data frames
            st.header('Response probability of basket and cluster combinations')
            st.write('In our approach, we perform a clustering step before basket analysis. This'
                     'allows us to stratify patients according to their omics, e.g. transcripts,'
                     'and use this information to help identify portions of the baskets that are'
                     'responsive to treatment.')

            basket_coords, cluster_coords = get_coords(save_data)
            basket_coords = basket_coords.tolist()
            cluster_coords = cluster_coords.tolist()
            total = len(basket_coords) * len(cluster_coords)
            selected_percent = int(0.25 * total)

            # show most responsive combinations in the left column
            col1, col2 = st.columns(2)
            with col1:
                st.subheader('Most responsive combinations')
                top_N = selected_percent # top 25%
                top_df = find_top_k_indices(save_data, top_N)
                top_df.drop(columns=['basket_idx', 'cluster_idx'], inplace=True)
                filter_top_df = st.checkbox('Filter empty?', key='filter_top_df')
                if filter_top_df:
                    top_df = top_df[top_df['count'] > 0]
                st.dataframe(top_df)

            # show least responsive combinations in the right column
            with col2:
                st.subheader('Least responsive combinations')
                bottom_N = selected_percent # bottom 25%
                bottom_df = find_bottom_k_indices(save_data, bottom_N)
                bottom_df.drop(columns=['basket_idx', 'cluster_idx'], inplace=True)
                filter_bottom_df = st.checkbox('Filter empty?', key='filter_bottom_df')
                if filter_bottom_df:
                    bottom_df = bottom_df[bottom_df['count'] > 0]
                st.dataframe(bottom_df)

            # create two dropdowns to select basket and cluster coords
            st.header('Basket and cluster interactions')
            st.write('Select a combination of basket and cluster below to examine it further')
            col1, col2 = st.columns(2)
            with col1:
                query_basket = st.selectbox('Basket:', options=basket_coords)
            with col2:
                query_cluster = int(st.selectbox('Cluster:', options=cluster_coords))

            y_highlight = basket_coords.index(query_basket) if query_basket else None
            x_highlight = cluster_coords.index(query_cluster) if query_cluster else None

            # plot overall heatmap
            inferred_df = get_basket_cluster_prob_df(save_data)
            fig = plot_basket_cluster_heatmap(inferred_df, x_highlight=x_highlight,
                                              y_highlight=y_highlight, return_fig=True)
            st.pyplot(fig)

            st.subheader('Member samples')
            col1, col2 = st.columns(2)
            with col1:
                st.write('The following is a list of samples found in this basket and '
                         'cluster combination.')
                selected_df = select_partition(save_data, query_basket, query_cluster)
                selected_df.drop(columns=['basket_number', 'cluster_number'], inplace=True)
                st.dataframe(selected_df)
            with col2:
                st.write('The following is a count of samples found in this basket and '
                         'cluster combination that are responsive (1) or non-responsive (0) '
                         'to the treatment.')
                fig = plot_responsive_count(selected_df, return_fig=True)
                st.pyplot(fig)

            member_df = get_member_expression(selected_df, save_data)
            if len(member_df) > 0:

                st.subheader('Samples PCA')
                st.write('The following is a PCA of transcript expression of samples '
                         'found in this basket and cluster combination.')
                try:
                    pc1, pc2, fig = plot_PCA(member_df, hue=selected_df['responsive'],
                                             n_components=2, return_fig=True)
                    st.pyplot(fig)
                except ValueError:
                    pass

                st.subheader('Significant transcripts')
                st.write('The following is a list of transcripts in this basket & cluster combination '
                         'whose mean expression levels are significantly different from the '
                         'entire dataset')
                all_expr_df = df_diff(member_df, save_data['expr_df_selected'])
                test_df = ttest_dataframe(member_df, all_expr_df, only_significant=True)
                st.dataframe(test_df)

                query_transcript = st.selectbox('Select transcript to examine further:',
                                                options=test_df.index.values)
                if query_transcript is not None:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write('You have selected ' + query_transcript)
                    with col2:
                        fig = plot_expression_boxplot(query_transcript, member_df, all_expr_df,
                                                      return_fig=True)
                        st.pyplot(fig)

app()
