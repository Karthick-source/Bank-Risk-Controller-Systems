import streamlit as st
import pandas as pd
import pickle
import numpy as np
from streamlit_option_menu import option_menu
import plotly.express as px
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
import pdfplumber
import os

# Add CSS for enhanced visuals
st.markdown("""
    <style>
        /* General Page Styling */
        body {
            font-family: 'Segoe UI', sans-serif;
            background-color: #f9f9f9;
        }
        
        /* Sidebar Styling */
        .css-1d391kg {
            background-color: #ffffff !important;
            border-right: 2px solid #007BFF;
        }

        /* Sidebar Options Styling */
        div[data-testid="stSidebar"] .css-1lcbmhc {
            padding: 10px;
            border-radius: 8px;
        }
        div[data-testid="stSidebar"] .css-1lcbmhc:hover {
            background-color: #e6f2ff;
            box-shadow: 0px 4px 6px rgba(0, 123, 255, 0.2);
        }

        /* Header Styling */
        h1, h2, h3 {
            color: #2e5984;
            font-weight: bold;
        }

        /* Button Styling */
        div.stButton > button {
            background-color: #007BFF;
            color: white;
            padding: 10px 20px;
            border-radius: 8px;
            font-size: 16px;
            font-weight: bold;
            border: none;
            transition: background-color 0.3s ease, transform 0.2s ease;
        }
        div.stButton > button:hover {
            background-color: #0056b3;
            transform: scale(1.05);
        }

        /* Dataframe Styling */
        .dataframe {
            border: 1px solid #dee2e6;
            border-radius: 8px;
            background-color: #ffffff;
            padding: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }

        /* File Uploader Styling */
        div[data-testid="stFileUploadDropzone"] {
            border: 2px dashed #007BFF !important;
            border-radius: 8px;
            background-color: #f7faff !important;
            padding: 10px;
        }

        /* Plot Titles */
        .plotly-title {
            font-size: 18px;
            font-weight: bold;
            color: #333333;
        }

        /* Cards for Metrics */
        .metric-card {
            background-color: #ffffff;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0px 2px 6px rgba(0, 0, 0, 0.1);
            margin: 10px 0;
            text-align: center;
        }
        .metric-card h3 {
            color: #007BFF;
            margin: 0;
        }
    </style>
""", unsafe_allow_html=True)


# Layout with Columns for Visual Separation
st.title("Bank Loan Clients Dashboard")

# Sidebar menu
with st.sidebar:
    menu = option_menu(
        menu_title="Navigation",
        options=["Data Display", "Visualization", "Prediction", "Bank Chatbot"],
        styles={
            "nav-link": {"font-size": "18px", "font-family": "Segoe UI"},
            "nav-link-selected": {"background-color": "#007BFF", "color": "white"}
        }
    )

    
#load dataset



with open('xg.pkl', 'rb') as file:
    model=pickle.load(file)

with open('label_encoder.pkl', 'rb') as en:
    model=pickle.load(en)

# Function to validate the model
def validate_model(model):
    if not hasattr(model, 'predict'):
        st.error("Invalid model. Please load the correct machine learning model.")
        return False
    return True


if menu =="Data Display":
    st.header("📊 Data Display")
    df=pd.read_csv("final.csv")
    metrics = {
        'Model': ['Decision Tree','Random Forest Classifier','Gradient Boosting Classifier','XGBoost'],
        'Accuracy': [68.34,66.32 ,69.16,94.36],
        'Precision': [65.64,66.47,69.47,92.84],
        'Recall': [77.0,65.87,0.9481,96.14],
        'F1 Score': [70.87,66.17,68.4,94.46],
        'ROC AUC': [68.34,66.32,69.16,94.36],

        'Confusion Matrix': [
            '[[139657  94381], [53857 180293]]',
            '[[156254  77784], [79918 154232]]',
            '[[163656  70382], [74003 160147]]',
            '[[216679  17359], [9029 225121]]']
    }
    metrics_df=pd.DataFrame(metrics)
    st.dataframe(metrics_df)
    st.dataframe(df.head(20))

elif menu == "Visualization":
    data = pd.read_csv("data_cleaned.csv")
    st.header("📈 Data Visualization")
    
    # Process data for visualization
    gender_count = data["CODE_GENDER"].value_counts().reset_index()
    gender_count.columns = ["CODE_GENDER", "count"]
    colors = ["#ffe5ff", "#00c89b"]
    
    # Create the Plotly figure for CODE_GENDER
    fig = px.bar(
        gender_count,
        x="CODE_GENDER",
        y="count",
        title="Gender Count",
        color="CODE_GENDER",
        color_discrete_sequence=colors
    )
    
    # Display the figure in Streamlit
    st.plotly_chart(fig)
    
    # Define the function to plot distributions for categorical columns
    def plot(col, colors):
        # Generate the value counts for the column
        df = data[col].value_counts().reset_index()
        df.columns = [col, "count"]
        
        # Create a Plotly bar chart
        fig = px.bar(
            df,
            x=col,
            y="count",
            title=f"Distribution of {col}",
            color=col,
            color_discrete_sequence=colors
        )
        
        # Display the figure in Streamlit
        st.plotly_chart(fig)
    
    # Get all categorical columns
    cat_col = data.select_dtypes(include="object").columns
    
    # Iterate over categorical columns and plot, excluding "CODE_GENDER"
    for col in cat_col:
        if col != "CODE_GENDER":  # Skip "CODE_GENDER" to avoid duplicate plot
            plot(col, colors)
    total_sum=data[["CODE_GENDER","AMT_INCOME_TOTAL"]].groupby("CODE_GENDER").agg(
    Total_income =("AMT_INCOME_TOTAL","sum"),
    Gender_count=("CODE_GENDER","count")
     ).reset_index()
    fig = px.bar(
        total_sum,
        x="CODE_GENDER",
        y="Total_income",
        title="Total Income and Gender Count by Gender",
        labels={"Total_income": "Total Income"},
        text="Total_income",  # Display the exact income values on bars
        color="CODE_GENDER",
        color_discrete_sequence=["#ffe5ff", "#00c89b"]
    )
    
    
    # Display the chart in Streamlit
    st.plotly_chart(fig)	



elif menu== "Prediction":
    st.header("🔮 Predict Loan Risks")
    df1 = pd.read_csv("final.csv")
    st.subheader("Enter the Features for Prediction")
    EXT_SOURCE_2 = st.number_input("EXT_SOURCE_2: ")
    EXT_SOURCE_3 = st.number_input("EXT_SOURCE_3: ")
    AMT_CREDIT_x = st.number_input("AMT_CREDIT_x: ")
    DAYS_EMPLOYED =st.number_input("DAYS_EMPLOYED: ",min_value=-15000, max_value=-198, value=-200)
    AMT_INCOME_TOTAL=st.number_input("AMT_INCOME_TOTAL: ")
    AMT_CREDIT_y=st.number_input("AMT_CREDIT_y: ")
    OCCUPATION_TYPE=st.number_input("OCCUPATION_TYPE: ",min_value=1, max_value=18, value=2)
    DAYS_LAST_DUE=st.number_input("DAYS_LAST_DUE: ",min_value=-2889, max_value=-2, value=-239)
    CNT_FAM_MEMBERS=st.number_input("CNT_FAM_MEMBERS: ",min_value=1, max_value=20, value=3)
    AMT_REQ_CREDIT_BUREAU_MON=st.number_input("AMT_REQ_CREDIT_BUREAU_MON: ",min_value=0, max_value=23, value=1)
    FLAG_OWN_REALTY=st.selectbox("FLAG_OWN_REALTY :",[1,0])
    AGE_GROUP=st.selectbox("AGE_GROUP :",[1,0,2])
    NAME_INCOME_TYPE=st.selectbox("NAME_INCOME_TYPE :",[6,0,2,3,5,4,1])
    NAME_CASH_LOAN_PURPOSE=st.number_input("NAME_CASH_LOAN_PURPOSE: ",min_value=0, max_value=24, value=23)
    FLAG_OWN_CAR=st.selectbox("FLAG_OWN_CAR:",[0,1])

#creating numericalinput feature for prediction
    numerical_features=pd.DataFrame({
        "EXT_SOURCE_2":[EXT_SOURCE_2],
        "EXT_SOURCE_3":[EXT_SOURCE_3],
        "AMT_CREDIT_x":[AMT_CREDIT_x],
        "DAYS_EMPLOYED":[DAYS_EMPLOYED],
        "AMT_INCOME_TOTAL":[AMT_INCOME_TOTAL],
        "AMT_CREDIT_y":[AMT_CREDIT_y],
        "DAYS_LAST_DUE":[DAYS_LAST_DUE],
        "CNT_FAM_MEMBERS":[CNT_FAM_MEMBERS],
        "AMT_REQ_CREDIT_BUREAU_MON":[AMT_REQ_CREDIT_BUREAU_MON]
        
    })
#Categorical input data
    categorical_features=pd.DataFrame({
        "OCCUPATION_TYPE":[OCCUPATION_TYPE],
        "FLAG_OWN_REALTY":[FLAG_OWN_REALTY],
        "AGE_GROUP":[AGE_GROUP],
        "NAME_INCOME_TYPE":[NAME_INCOME_TYPE],
        "NAME_CASH_LOAN_PURPOSE":[NAME_CASH_LOAN_PURPOSE],
        "FLAG_OWN_CAR":[FLAG_OWN_CAR]
    })

	
     # Load encoders and transform categorical features
     # Load encoders and transform categorical features
    with open('label_encoder.pkl', 'rb') as en:
        encoder = pickle.load(en)
    with open('xgb.pkl', 'rb') as file:
        model=pickle.load(file)
    st.write(f"Model type: {type(model)}")



    # Function to validate the model
    def validate_model(model):
        if not hasattr(model, 'predict'):
            st.error("Invalid model. Please load the correct machine learning model.")
            return False
        return True

    for column in categorical_features.columns:
        try:
            # Transform the data using the corresponding encoder
            categorical_features[column] = encoder[column].transform(categorical_features[column])
        except ValueError:
            # Append a default "unknown" class to the encoder's classes
            encoder[column].classes_ = np.append(encoder[column].classes_, "unknown")
            
            # Replace unseen labels with "unknown"
            categorical_features[column] = categorical_features[column].apply(
                lambda x: x if x in encoder[column].classes_ else "unknown"
            )
            
            # Transform again silently
            categorical_features[column] = encoder[column].transform(categorical_features[column])

    # Combine numerical and encoded categorical features
    input_data = pd.concat([numerical_features.reset_index(drop=True),
                            categorical_features.reset_index(drop=True)], axis=1)

    # Ensure the input data matches the model’s expected feature order
    expected_feature_names = [
        "EXT_SOURCE_2", "EXT_SOURCE_3", "AMT_CREDIT_x", "DAYS_EMPLOYED", 
        "AMT_INCOME_TOTAL", "AMT_CREDIT_y", "OCCUPATION_TYPE", 
        "DAYS_LAST_DUE", "CNT_FAM_MEMBERS", "AMT_REQ_CREDIT_BUREAU_MON", 
        "FLAG_OWN_REALTY", "AGE_GROUP", "NAME_INCOME_TYPE", 
        "NAME_CASH_LOAN_PURPOSE", "FLAG_OWN_CAR"
    ]
    input_data = input_data[expected_feature_names]

    # Prediction
    if st.button('Calculate Prediction'):
        with st.spinner('Predicting...'):
            try:
                # Get the prediction probabilities
                predicted_prob = model.predict_proba(input_data)[:, 1]  # Probability of class 1
                
                # Define the decision threshold
                threshold = 0.5  # You can adjust this based on model performance
                
                # Display the result based on the threshold
                if predicted_prob[0] >= threshold:
                    st.success(f"The client is at **risk of payment difficulties** with a probability of {predicted_prob[0]:.2f}.")
                else:
                    st.success(f"The client is **not at risk of payment difficulties** with a probability of {predicted_prob[0]:.2f}.")
            except ValueError as e:
                st.error(f"Prediction failed: {e}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")




# Define the PromptTemplate
template = """
You are an assistant that helps users with banking-related questions based on a provided context.

Please provide concise, accurate, and clear answers. Avoid repetition.
If the context is insufficient, let the user know more information is needed.

Context:
{context}

User Question: 
{question}

Provide the most relevant information to answer the question based on the context provided.
"""
prompt = PromptTemplate.from_template(template)

# Function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    full_text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                full_text += page_text
    return full_text

# Function to clean LLM response
def clean_response(response_text):
    words = response_text.split()
    seen = set()
    filtered_words = []
    for word in words:
        if word not in seen:
            seen.add(word)
            filtered_words.append(word)
    return " ".join(filtered_words)

# Streamlit interface
st.title("Bank Chatbot")
st.write("Ask questions related to banking.")

uploaded_file = st.file_uploader("Upload a Banking-related PDF", type="pdf")

# Add a debug toggle
show_debug = st.checkbox("Show Debug Outputs", value=False)

# If a file is uploaded
if uploaded_file is not None:
    # Extract text from the uploaded PDF
    full_text = extract_text_from_pdf(uploaded_file)

    # Display the extracted content for user reference (show a snippet)
    st.text_area("Extracted PDF Content", full_text[:1000], height=300)  # Show only the first 1000 characters

    # Split the text into chunks for processing
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    documents = [Document(page_content=chunk) for chunk in text_splitter.split_text(full_text)]

    # Debug: Display the first 3 chunks for user reference if debug is enabled
    if show_debug:
        st.write("Documents in vector DB:", [doc.page_content[:100] for doc in documents[:3]])

    # Create embeddings and vector database
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_db = FAISS.from_documents(documents, embeddings)
    except Exception as e:
        st.error(f"Error loading embeddings or creating vector DB: {e}")
        vector_db = None

    # Load Chat Model
    model_file = r"C:\Users\admin\python\Bank Risk Controller Systems\llama"  # Update with actual model file path
    if os.path.exists(model_file):
        llm = CTransformers(
            model=model_file,
            model_type="llama",
            config={"max_new_tokens": 500, "temperature": 0.01}
        )
    else:
        st.error(f"Model file not found at {model_file}")
        llm = None

    # Create the RetrievalQA model
    if llm and vector_db:
        chatbox_model = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        )

        # User input for questions
        user_question = st.text_input("Ask a question about banking:")

        # Special handling for "Hi" or "Hello" responses
        if user_question.lower() in ["hi", "hello"]:
            st.success("Hello, I'm a bank chatbot, here to answer your queries.")
        else:
            if st.button("Get Answer"):
                with st.spinner("Fetching the answer..."):
                    try:
                        # Retrieve relevant documents and debug if enabled
                        retrieved_docs = vector_db.similarity_search(user_question, k=3)
                        if show_debug:
                            st.write("Debug: Retrieved Documents", [doc.page_content[:200] for doc in retrieved_docs])

                        # Pass the query and retrieved context to the LLM
                        response = chatbox_model({"query": user_question})
                        if show_debug:
                            st.write("Debug: Raw response:", response)

                        # Extract and clean the answer
                        if response.get("result"):
                            answer = clean_response(response.get("result"))
                        else:
                            answer = "I'm unable to find relevant information about this question in the uploaded document."

                        st.success(f"Answer: {answer}")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
    else:
        st.warning("Could not initialize the chatbot. Ensure embeddings and model files are set up correctly.")
else:
    st.warning("Please upload a PDF to get started.")