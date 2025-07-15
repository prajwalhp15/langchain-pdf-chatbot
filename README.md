# LangChain PDF Chatbot

This project is an AI-powered chatbot that allows users to upload any PDF (such as a research paper or report) and ask questions about its content. It uses LangChain and Hugging Face models to perform retrieval-based question answering.

The goal is to help students, researchers, and professionals interact with long documents using natural language queries.

---

## Features

- Upload any PDF file for processing
- Ask natural language questions about the document
- Uses document chunking and vector embedding for retrieval
- Powered by Hugging Face LLM (Flan-T5) and LangChain framework
- Local and free (no OpenAI key required)

---

## Project Structure

langchain-pdf-chatbot/
├── app.py # Streamlit app (optional, if used)
├── main.py # Core chatbot script
├── paper.pdf # Sample PDF (you can replace with your own)
├── requirements.txt # Python dependencies
└── README.md # Project overview and usage

yaml
Copy
Edit

---

## How It Works

1. Loads the PDF file using LangChain's `PyPDFLoader`
2. Splits the text into manageable chunks
3. Embeds the chunks using `HuggingFaceEmbeddings` (`all-MiniLM-L6-v2`)
4. Stores the chunks in a vector database using Chroma
5. Accepts user questions and retrieves relevant content
6. Uses a Hugging Face LLM (`flan-t5-base`) to generate answers

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/prajwalhp15/langchain-pdf-chatbot.git
cd langchain-pdf-chatbot
2. (Optional) Create a Virtual Environment
bash
Copy
Edit
python -m venv venv
venv\Scripts\activate  # On Windows
3. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
4. Add Your Hugging Face Token
Create a .env file in the project root with this line:

ini
Copy
Edit
HUGGINGFACEHUB_API_TOKEN=your_token_here
You can generate your token at: https://huggingface.co/settings/tokens
Use "Read" permission.

5. Run the Chatbot
bash
Copy
Edit
python main.py
Then ask questions like:

"What is the title of the paper?"

"Summarize the abstract."

"List the key contributions."

Example Output
vbnet
Copy
Edit
You: What is the title of the paper?
Bot: Show, attend and tell: Neural image caption generation with visual attention
License
This project is open-source and available for non-commercial and educational use.

Credits
Built with:

LangChain

Hugging Face Transformers

Streamlit (optional)

ChromaDB

PyMuPDF for PDF parsing
