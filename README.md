# Ai-Semantic-Search-App
Semantic Search App using Langchain, FAISS, OpenAI, Streamlit.


Semantic Search App designed for effortless information retrieval. Users can input article URLs and ask questions to receive relevant insights from the URL.

## Features

- Load URLs or upload text files containing URLs to fetch article content.
- Process article content through LangChain's UnstructuredURL Loader.
- Construct an embedding vector using OpenAI's embeddings and leverage FAISS, a powerful similarity search library, to enable swift and effective retrieval of relevant information.
- Interact with the LLM by inputting queries and receiving answers along with source URLs.


## Installation

1.Clone this repository to your local machine using:

```bash
  git clone https://github.com/tjdharani/Ai-Semantic-Search-App.git
```
2.Navigate to the project directory:

```bash
  cd Ai-Semantic-Search-App
```
3.Install the required dependencies using pip:

```bash
  pip install -r requirements.txt
```
4.Set up your OpenAI API key by creating a .env file in the project root and adding your API

```bash
  OPENAI_API_KEY=your_api_key_her
```

## Usage/Examples

1.Run the Streamlit app by executing:
```bash
streamlit run app.py

```

2.The web app will open in your browser.

- On the sidebar, you can input URLs directly.

- Initiate the data loading and processing by clicking "Process URLs."

- Observe the system as it performs text splitting, generates embedding vectors, and efficiently indexes them using FAISS.

- The embeddings will be stored and indexed using FAISS, enhancing retrieval speed.

- The FAISS index will be saved in a local file path in pickle format for future use.

- You can now ask a question and get the answer based on those URL articles.


## Project Structure

- app.py: The main Streamlit application script.
- requirements.txt: A list of required Python packages for the project.
- data.pkl: Pickle File for storing FAISS index. 
