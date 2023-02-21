import requests
from bs4 import BeautifulSoup
import streamlit as st
import pandas as pd
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
import trafilatura
from trafilatura import fetch_url, extract
from trafilatura.settings import use_config
import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import streamlit.components.v1 as components
from pivottablejs import pivot_ui



tab1, tab2,tab3 = st.tabs(["Data extraction", "Model","Others"])

import streamlit as st

with tab1:
    uploaded_file1 = st.file_uploader("Choose Related Terms")
    if uploaded_file1 is not None:
        uploaded_file1 = pd.read_csv(uploaded_file1)

    uploaded_file2 = st.file_uploader("Choose Search Suggestions")
    if uploaded_file2 is not None:
        uploaded_file2 = pd.read_csv(uploaded_file2)

        uploaded_file = pd.concat([uploaded_file1[['Keyword', 'Difficulty','Volume','SERP Features']], uploaded_file2[['Keyword', 'Difficulty','Volume','SERP Features']]])
        uploaded_file = uploaded_file.drop_duplicates(subset=['Keyword'])
        uploaded_file = uploaded_file.sort_values(by = 'Volume',ascending=False)
        uploaded_file['Difficulty'] = uploaded_file['Difficulty'].fillna(0)
        uploaded_file['Volume'] = uploaded_file['Volume'].fillna(0)
        uploaded_file['SERP Features'] = uploaded_file['SERP Features'].fillna('not found')
        uploaded_file = uploaded_file.reset_index(drop=True)
        st.write(uploaded_file)

        docs = uploaded_file["Keyword"].to_list()
        # best_model = SentenceTransformer("all-mpnet-base-v2")
        fast_model = SentenceTransformer("all-MiniLM-L6-v2")

        # vectorizer_model = CountVectorizer(ngram_range=(1,3), stop_words="english")
        # umap_model = UMAP(n_neighbors=5)
        # hdbscan_model = HDBSCAN(min_cluster_size=10, metric='euclidean', prediction_data=True)
        # cluster_model = KMeans(n_clusters=50)

        # embeddings = fast_model.encode(docs, show_progress_bar=False) 

        topic_model = BERTopic(embedding_model=fast_model,
                               # vectorizer_model=vectorizer_model,
                              #  umap_model = umap_model,
                               # hdbscan_model=cluster_model,
                               language="english",
                              #  n_gram_range=(1, 2),
                               min_topic_size=2,
                               # verbose=True,
                               calculate_probabilities=False)

        # topics, probabilities = topic_model.fit(docs,y=Pre_Category) # for supervised topic modelling

        topics = topic_model.fit_transform(docs)

        #assign topics back to table
        tt = pd.DataFrame(topic_model.topic_labels_,index=[0]).transpose()
        tt = tt.reset_index()
        tt.columns = ['index','Group']
        tt2 = pd.DataFrame(topic_model.topics_)
        tt2.columns = ['index']
        Left_join = pd.merge(tt, 
                             tt2, 
                             on ='index', 
                             how ='right')
        result = pd.concat([Left_join, uploaded_file],axis=1)
        result2 = result.sort_values('Volume',ascending=False).drop_duplicates(['Group'],keep='first')
        final = pd.merge(result,
                         result2[['Group','Keyword']], 
                             on ='Group', 
                             how ='left')

        final.rename(columns={'Keyword_x': 'Keyword', 'Keyword_y': 'Created_Group'}, inplace=True)
        final.sort_values(by=["Group","Volume"],ascending=False,inplace=True)
        t = pivot_ui(final)
        with open(t.src) as t:
            components.html(t.read(), width=900, height=1000, scrolling=True)

        featured_snippet = final[final['SERP Features'].str.contains('Featured')]
        featured_snippet.sort_values(by=["Volume","Group"],ascending=False,inplace=True)
        featured_snippet=featured_snippet[['Keyword','Volume','Difficulty','SERP Features','Created_Group']]
        st.caption("Featured Snippet Opportunities")
        st.write(featured_snippet)



with tab2:
    
    # URL of the webpage to scrape
    os.environ["OPENAI_API_KEY"] = st.text_input("Enter OpenAI API Key")
    url = st.text_input("Enter URL")
    article = st.text_input("Enter Keyword")
    agree = st.checkbox('Start')
    if agree:
    # Make a GET request to the webpage
        response = requests.get(url)

        # Parse the HTML content using Beautiful Soup
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find all the h2 tags on the page
        h2_tags = soup.find_all('h2', class_=lambda x: x != 'entry-title')

        # List to store the scraped headings
        headings = []

        # Loop through each h2 tag
        for h2 in h2_tags:
            # Get the text of the h2 tag
            h2_text = h2.get_text().strip()
            
            # Find the next siblings of the h2 tag until the next h2 tag
            siblings = h2.find_next_siblings()
            headings.append(h2_text)
            for sibling in siblings:
                
                # If a new h2 tag is found, break out of the loop
                if sibling.name == 'h2':
                    break
                # If a h3 or h4 tag is found, append it to the list of headings
                elif sibling.name in ['h3', 'h4']:
                    headings.append(sibling.get_text().strip())

        # Print the list of scraped headings
        headings = ', '.join(headings)

        template = """I want to rewrite an article about {article_name}.
        The current outline of the article has been mentioned below. I want to add to the current outline, new topics that can be helpful for the readers of the article. 
        Build upon the current outline and give me a new outline with the new topics included. Provide the outline in a readable format and use new lines if needed.\n
        Outline:{article_outline} """
        prompt = PromptTemplate(template=template, input_variables=["article_name","article_outline"])
        llm_chain = LLMChain(prompt=prompt, llm=OpenAI(temperature=0,max_tokens=512), verbose=True)

        st.write(response)
        st.write(llm_chain.predict(article_outline=headings,article_name=article))

        newconfig = use_config()
        newconfig.set("DEFAULT", "EXTRACTION_TIMEOUT", "0")

        downloaded = trafilatura.fetch_url(url)
        result = extract(downloaded,config=newconfig)

        template2 = """I want to generate questions and answers for an article about {article_name}.
        Using the article context below, generate questions and answer each question from the context below.
        Each Question should be relevant to the context of the article. Leave a line between the question and answer.\n

        Context:{context}
        Questions and Answers: """
        prompt2 = PromptTemplate(template=template2, input_variables=["article_name","context"])
        llm_chain2 = LLMChain(prompt=prompt2, llm=OpenAI(temperature=0,max_tokens=1024), verbose=True)

        st.write(prompt2)
        st.write(llm_chain2.predict(context=result,article_name=article))
