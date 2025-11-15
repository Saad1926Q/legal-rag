# Legal RAG 

## Project Overview

So I am implementing a multi-agent RAG system focused on Indian Criminal Law. The system uses multiple AI agents that work together to retrieve relevant legal information and generate accurate responses with proper citations.

## Documents

The documents I have added are organized into three categories:

### Statutes
1. The Indian Penal Code, 1860
2. The Code of Criminal Procedure, 1973
3. The Indian Evidence Act, 1872
4. The Protection of Children from Sexual Offences Act, 2012 (POCSO)
5. The Scheduled Castes and Scheduled Tribes (Prevention of Atrocities) Act, 1989

### Case Laws
1. Bachan Singh v. State of Punjab (1980) - Landmark death penalty judgement
2. Maneka Gandhi v. Union of India - Fundamental rights interpretation
3. Arnesh Kumar v. State of Bihar - Arrest guidelines
4. State of Maharashtra v. Madhukar Narayan Mardikar - Burden of proof
5. Vikas Yadav v. State of U.P. - Witness protection

### Regulations
1. Model Prison Manual for the Superintendence and Management of Prisons in India
2. The Model Police Act, 2006

## Corpus Statistics
- Total documents: 13
- Total pages processed: 1,359
- Chunks indexed: 5,477
- Vector database size: 41MB

## Running the Application

### Streamlit UI (Recommended)
```bash
# Activate virtual environment
source .venv/bin/activate

# Run the Streamlit app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### CLI Testing
```bash
# Activate virtual environment
source .venv/bin/activate

# Run the retrieval agent directly
python src/agents/retrieval_agent.py
```