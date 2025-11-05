{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "940e3e9e-1b41-4233-8a93-2646294b5645",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "from langchain.llms import OpenAI\n",
    "from langchain.text_splitter import CharacterTextSplitter\n",
    "from langchain.embeddings import OpenAIEmbeddings\n",
    "from langchain.vectorstores import Chroma\n",
    "from langchain.chains import RetrievalQA"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "819386d8-12dd-4a9e-96fe-f0a8c50ca175",
   "metadata": {},
   "outputs": [],
   "source": [
    "def generate_response(uploaded_file, openai_api_key, query_text):\n",
    "    # Load document if file is uploaded\n",
    "    if uploaded_file is not None:\n",
    "        documents = [uploaded_file.read().decode()]\n",
    "        \n",
    "       # Split documents into chunks\n",
    "        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)\n",
    "        texts = text_splitter.create_documents(documents)\n",
    "        \n",
    "       # Select embeddings\n",
    "        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)\n",
    "       \n",
    "        # Create a vectorstore from documents\n",
    "        db = Chroma.from_documents(texts, embeddings)\n",
    "       \n",
    "       # Create retriever interface\n",
    "        retriever = db.as_retriever()\n",
    "\n",
    "       # Create QA chain\n",
    "        qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=openai_api_key), chain_type='stuff', retriever=retriever)\n",
    "return qa.run(query_text)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7071f3ec-a554-4e5b-94ac-1e8e461cf265",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Page title\n",
    "st.set_page_config(page_title='🦜🔗 Ask the Doc App')\n",
    "st.title('🦜🔗 Ask the Doc App')\n",
    "\n",
    "# File upload\n",
    "uploaded_file = st.file_uploader('Upload an article', type='txt')\n",
    "\n",
    "# Query text\n",
    "query_text = st.text_input('Enter your question:', placeholder = 'Please provide a short summary.', disabled=not uploaded_file)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4a9ccc26-2ef7-41e6-9c7c-6bef737731a7",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Form input and query\n",
    "result = []\n",
    "with st.form('myform', clear_on_submit=True):\n",
    "    openai_api_key = st.text_input('OpenAI API Key', type='password', disabled=not (uploaded_file and query_text))\n",
    "    submitted = st.form_submit_button('Submit', disabled=not(uploaded_file and query_text))\n",
    "    if submitted and openai_api_key.startswith('sk-'):\n",
    "        with st.spinner('Calculating...'):\n",
    "            response = generate_response(uploaded_file, openai_api_key, query_text)\n",
    "            result.append(response)\n",
    "            del openai_api_key\n",
    "\n",
    "if len(result):\n",
    "    st.info(response)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python (homl3)",
   "language": "python",
   "name": "homl3"
  },
  "language_info": {
   "name": ""
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
