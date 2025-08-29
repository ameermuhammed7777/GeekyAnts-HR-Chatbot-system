Project Structure

Directory structure:
└── ameermuhammed7777-geekyants-hr-chatbot-system/
    ├── backend_functions.py
    ├── model.py
    ├── stmlit.py
    └── utils/
        ├── employee_table.txt
        ├── employees.yaml
        └── requirements.txt


stmlit.py : Streamlit app (main entry point for chatbot UI
backend_functions.py : Core backend logic and utility functions
model.py : FastAPI service for embeddings using 'all-MiniLM-L6-v2'
utils/employee_table.txt :  Sample employee data
requirements.txt : Project dependencies


## ⚙️ Setup Instructions  


# 1. Clone the repository  
git clone https://github.com/ameermuhammed7777/geekyants-hr-chatbot-system.git
cd ameermuhammed7777-geekyants-hr-chatbot-system


# 2. Create a virtual environment  
python -m venv venv


# 3. Activate the virtual environment  
source venv/bin/activate 


# 4. Install dependencies 
pip install -r utils/requirements.txt


# 5. Start the FastAPI server (runs the embeddings model service)
uvicorn model:app --reload --host 0.0.0.0 --port 9999


# 6. Run the Streamlit app (chatbot user interface)
streamlit run stmlit.py


