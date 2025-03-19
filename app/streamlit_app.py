import streamlit as st
import requests

st.title("Chat With Repository")

# URL input field
url = st.text_input("Enter the repository URL:")

# Create knowledge base button
if st.button("Create Knowledge Base"):
    if url:
        st.write("Creating knowledge base...")
        response = requests.post(
            "http://localhost:8000/create_knowledge_base/", json={"url": url}
        )
        if response.status_code == 200:
            st.write("Knowledge base created successfully!")
        else:
            st.write(f"Status Code: {response.status_code}")
            st.write(f"Response Content: {response.content}")
            try:
                error_detail = response.json().get("detail")
            except requests.exceptions.JSONDecodeError:
                error_detail = response.text  # Fallback to raw response text
            st.write(f"Error: {error_detail}")
    else:
        st.write("Please enter a repository URL.")

# Query input field
query = st.text_input("Enter your query:")

# Get answer button
if st.button("Get Answer"):
    if query:
        response = requests.post(
            "http://localhost:8000/get_answer/", json={"query": query}
        )
        if response.status_code == 200:
            try:
                answer = response.json().get("answer")
            except requests.exceptions.JSONDecodeError:
                answer = response.text  # Fallback to raw response text
            st.write(answer)
        else:
            st.write(f"Status Code: {response.status_code}")
            st.write(f"Response Content: {response.content}")
            try:
                error_detail = response.json().get("detail")
            except requests.exceptions.JSONDecodeError:
                error_detail = response.text  # Fallback to raw response text
            st.write(f"Error: {error_detail}")
    else:
        st.write("Please enter a query.")
