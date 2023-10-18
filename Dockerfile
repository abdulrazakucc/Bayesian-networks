# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.11

WORKDIR /app
COPY . /app

# Install pip requirements
COPY requirements.txt .
RUN python -m pip install -r requirements.txt

EXPOSE 8501 

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
ENTRYPOINT ["streamlit", "run", "toy_bayesian_model.py", "--server.port=8501", "--server.address=0.0.0.0"]
