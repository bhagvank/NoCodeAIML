import streamlit as st
from pymilvus import connections, Collection,list_collections
import pandas as pd

with st.sidebar:
    connections.connect("default", host='localhost', port='19530')
    
    
st.title("📝 Milvius Database Viewer")
question = st.text_input(
    "Show Collections",
    placeholder="Show Collections"
)



    # Get the list of all collections
collections = list_collections()

    # Print the collection names
msgr = st.chat_message("assistant")

msgr.write("Collections in Milvus:")
st.chat_message("assistant").write( collections)
msg = collections
st.chat_message("assistant").write(msg)


for collection in collections:
    collection_name = collection
    
    print("collection_name",collection_name)

    # Load the collection
    collection = Collection(collection_name)
    
    schema = collection.schema
    
    print("Fields in collection:",schema.fields)
    
    for field in schema.fields:
        st.chat_message("assistant").write(f"Field name: {field.name}, Field type: {field.dtype}")
    
    fields = []
    
    for field in schema.fields:
        fields.append(field.name)

