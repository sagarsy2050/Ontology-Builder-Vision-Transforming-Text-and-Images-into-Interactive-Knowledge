@echo off
echo Setting up Ollama Ontology Vision Streamlit...

pip install -r requirements.txt
python -m spacy download en_core_web_sm

start /B ollama serve
timeout /t 5
ollama pull llama3.1:8b
ollama pull llava:13b

echo Setup complete! Run: streamlit run app_vision.py
pause