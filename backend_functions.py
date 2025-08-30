import numpy as np
import requests
import faiss
from dotenv import load_dotenv
import os
import google.generativeai as genai



load_dotenv("./utils/.env")

Geminii_API_Key = os.getenv("Geminii_API_Key")
Model = os.getenv("Model")

class functionality:
    def __init__(self):
        self.docs=[]

    def textload(self,path):
        with open(path, "r") as f:
            data =f.read()
        for doc in data.split("---"):
            doc = doc.strip()
            if doc:
                self.docs.append(doc)
        print("[LOG] Text Loading Completed")
        return self.docs

    

    def build_faiss_index(self, _docs):
        """Convert documents to embeddings and build FAISS index"""
        embeddings = []
        for doc in _docs:
            response = requests.post("http://localhost:9999/embed", json={"text": [doc]}).json()
            emb = np.array(response["array"]).reshape(response["shape"])[0]
            embeddings.append(emb)

        embeddings = np.array(embeddings).astype("float32")
        dimension = embeddings.shape[1]

        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        print("[LOG] Vectore Store Created")
        return index
    
    def doc_retrive(self,user_query,index,docs):
        response = requests.post("http://localhost:9999/embed", json={"text": [user_query]}).json()
        query_emb = np.array(response["array"]).reshape(response["shape"])[0].astype("float32")

        D, I = index.search(np.array([query_emb]), k=3)

        matched_docs = [docs[idx] for idx in I[0]]

        print("[LOG] Doc Retriever Completed")
        return matched_docs
    

    def QandA(self,user_query,matched_docs):
        prompt = f"""You are an HR assistant. The user asked: "{user_query}".

                        Your task:
                        - Use only the relevant information from the context (ignore unrelated details).
                        - Provide a clear and accurate answer to the query.
                        - If multiple candidates match, list them clearly.
                        - Format the response in a professional HR style.

                        Here are some examples for reference:

                        Example 1:
                        User: "I need someone experienced with machine learning for a healthcare project"
                        Response:
                        Based on your requirements for ML expertise in healthcare, I found 2 excellent candidates:

                        **Dr. Sarah Chen** – 6 years of ML experience, worked on 'Medical Diagnosis Platform' (X-ray analysis with computer vision).
                        Skills: TensorFlow, PyTorch, medical data processing. Availability: Immediate.

                        **Michael Rodriguez** – 4 years of ML experience, built 'Patient Risk Prediction System' (ensemble methods, HIPAA compliance).
                        Skills: scikit-learn, pandas, EHR data handling. Availability: Immediate.

                        Do not produce answers from the example given in case the particular field is not found.
                        Now answer the current query accurately using the given context.

                        context: 
                        {matched_docs}
                        """
        
        genai.configure(api_key=Geminii_API_Key)
        model = genai.GenerativeModel(Model)
        response = model.generate_content(prompt)

        print("[LOG] Response Generated")
        return response.candidates[0].content.parts[0].text






