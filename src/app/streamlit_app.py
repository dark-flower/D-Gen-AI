import streamlit as st
from backend.solution import answer_user_question







st.set_page_config(page_title="Title", layout="centered")


st.header('**Title**')
col1, col2 = st.columns(2, gap='large')
with col1:
    col1.write("**SubTitle**")
#with col2:
#    col2.write("**انا هنا للرد علي اسألتكم**")

st.divider()

with st.form("questions"):

    
    greetings_ph = st.empty()

    lang = st.radio("Please select the Model language:",
                    ["Arabic", "English"],
                    captions = ["Questions", 
                                "Questions"],
                                horizontal=True)

    question = st.text_input("Please write your question here: ", max_chars=300)
    # Every form must have a submit button.
    cols = st.columns(4, gap='large')
    with cols[-1]:
        submitted = cols[-1].form_submit_button("Submit")

    if submitted:

        with st.spinner('I am working on it...'):
            tgt_lang = 'ar' if lang == 'Arabic' else 'en'
            result = answer_user_question(question, tgt_lang)
        st.write(f"**Answer:**\n{result['response']}")
        
        with  st.expander("Show the response details!"):
            
            src_docs = ' '.join( x.page_content  for x in  result['source_documents'])

            st.write(f"**This text was used to find an answer:**\n\n{src_docs}")














