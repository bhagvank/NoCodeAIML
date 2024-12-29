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


collections = list_collections()

msgr = st.chat_message("assistant")

msgr.write("Collections in Milvus:")
st.chat_message("assistant").write( collections)
msg = collections
st.chat_message("assistant").write(msg)


for collection in collections:
    collection_name = collection
    
    print("collection_name",collection_name)

    collection = Collection(collection_name)
    
    schema = collection.schema
    
    print("Fields in collection:",schema.fields)
    
    for field in schema.fields:
        st.chat_message("assistant").write(f"Field name: {field.name}, Field type: {field.dtype}")
    
    fields = []
    
    for field in schema.fields:
        fields.append(field.name)

    results = collection.query(expr="id >=0", output_fields=fields) 

    df = pd.DataFrame(results)
    
    st.chat_message("assistant").write("data sample :")
    msgr.write(df.head(10))
