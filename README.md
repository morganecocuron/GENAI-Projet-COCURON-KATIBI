# GENAI-Projet-COCURON-KATIBI



**Group Members \& TD Group**



Group members: COCURON Morgane, KATIBI Maria

TD group number: ESILV A5 - CDOF2



**Project Overview**



This project implements a fully local LLM Council inspired by Andrej Karpathy’s concept.



Instead of relying on a single model, the system coordinates multiple local LLMs to answer a user query, review each other’s responses, and produce a final high-quality answer using a dedicated model called the Chairman.



The orchestration follows a 3-stage pipeline:



Stage 1: Council models generate independent answers.



Stage 2: Models review and rank anonymized responses.



Stage 3: The Chairman synthesizes the final response using answers + rankings.



The system runs on a distributed setup:



PC A (Orchestrator): FastAPI backend + council models



PC B (Chairman): Ollama server exposing only the Chairman model through REST API



**Setup and Installation Instructions**



*1) Requirements*



Python 3.10+



Ollama installed on both machines : https://ollama.com/download



Two machines connected on the same local network (for the demo)



*2) Install Ollama models* (open a command line window: type cmd in the search bar)



On PC A (Council models):



ollama pull llama3

ollama pull mistral

ollama pull qwen2.5



On PC B (Chairman model):



ollama pull llama3



*3) Backend setup (PC A)*



Open a terminal in VS Code at the root of the project, then:

cd backend



Create the virtual environment (required):

python -m venv venv



Activate the virtual environment (PowerShell):

.\\venv\\Scripts\\Activate.ps1



Install backend dependencies:

pip install fastapi uvicorn httpx python-dotenv



Create and configure the backend .env file:



COUNCIL\_BASE\_URL=http://localhost:11434

CHAIRMAN\_BASE\_URL=http://<PC\_B\_IP>:11434

COUNCIL\_MODELS=mistral,qwen2.5,llama3

CHAIRMAN\_MODEL=llama3



*4) Frontend setup (PC A)*



cd frontend

npm install



Configure the frontend .env file:

VITE\_API\_BASE=http://localhost:8000



**Instructions to Run the Demo**



*Step 1 — Start Ollama on PC B (Chairman)*



On PC B:

ollama serve



To get the IP address of PC B:

ipconfig



Use the IPv4 Address in your backend .env file:

CHAIRMAN\_BASE\_URL=http://IP\_PC\_B:11434



*Step 2 — Start Ollama on PC A (Council)*



On PC A:

ollama serve



*Step 3 — Start the FastAPI backend (PC A)*



Go back to the root of the project:

cd ..



Run the backend:

python -m uvicorn backend.main:app --reload --port 8001



*Step 4 — Start the frontend (PC A)*



cd frontend

npm run dev



*Step 5 — Run a test query*



Open the web interface and submit a question (example: “What is deep learning?”).



The system will automatically execute Stage 1 → Stage 2 → Stage 3 and display the final Chairman answer.

