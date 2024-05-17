import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
from PIL import Image, ImageTk
from PyPDF2 import PdfReader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
import os
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import OpenAI

os.environ["OPENAI_API_KEY"] = "sk-proj-5t1pOBdADM71U8aW1RuGT3BlbkFJDcrWpt3ZSqhr9ZlfuDQt"

# Provide the paths of PDF files.
pdf_paths = [
    r"C:\Users\NAHAS\Desktop\ChatBot\IPC part 2.pdf",
    r"C:\Users\NAHAS\Desktop\ChatBot\IPC part 3.pdf",
    r"C:\Users\NAHAS\Desktop\ChatBot\IPC Sections.pdf"
]

raw_texts = []

for path in pdf_paths:
    pdf_reader = PdfReader(path)
    raw_text = ''
    for page in pdf_reader.pages:
        content = page.extract_text()
        if content:
            raw_text += content
    raw_texts.append(raw_text)

# We need to split the text using Character Text Split such that it should not increase token size
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=800,
    chunk_overlap=200,
    length_function=len,
)

# Split and flatten texts
texts = [text_splitter.split_text(raw_text) for raw_text in raw_texts]
flattened_texts = [item for sublist in texts for item in sublist]

# Download embeddings from OpenAI
embeddings = OpenAIEmbeddings()

# Create FAISS index from the flattened texts
document_search = FAISS.from_texts(flattened_texts, embeddings)

chain = load_qa_chain(OpenAI(), chain_type="stuff")

def get_response():
    user_input = user_input_entry.get("1.0", tk.END).strip()
    bot_response = ""

    if user_input.lower() in ["hi", "hello", "hey", "hy"]:
        bot_response = "Hello, welcome to Legal Chatbot. How can I assist you today!"
    elif user_input.lower() in ["bye", "by", "thank you", "thanks"]:
        bot_response = "Bye, and have a good day!"
    else:
        if len(user_input) < 4:
            bot_response = "Please enter a valid question!"
        else:
            docs = document_search.similarity_search(user_input)
            bot_response = chain.run(input_documents=docs, question=user_input)

    chat_window.config(state=tk.NORMAL)
    chat_window.insert(tk.END, "You: " + user_input + "\n", "user")
    chat_window.insert(tk.END, "Bot: " + bot_response + "\n", "bot")
    chat_window.config(state=tk.DISABLED)
    user_input_entry.delete("1.0", tk.END)

root = tk.Tk()
root.title("Legal Chatbot")

notebook = ttk.Notebook(root)
notebook.pack(fill=tk.BOTH, expand=True)

home_frame = ttk.Frame(notebook)
notebook.add(home_frame, text="Legalbot")

image = Image.open(r"C:\Users\NAHAS\Desktop\ChatBot\image.jpeg")

max_width = 200
max_height = 100
image.thumbnail((max_width, max_height), Image.LANCZOS)

photo = ImageTk.PhotoImage(image)
image_label = tk.Label(home_frame, image=photo)
image_label.image = photo
image_label.pack(pady=20)

chat_window = scrolledtext.ScrolledText(home_frame, wrap=tk.WORD, state=tk.DISABLED, width=70, height=15, bg='bisque3')
chat_window.tag_configure("user", foreground="red")
chat_window.tag_configure("bot", foreground="black")
chat_window.pack(pady=20)

user_input_entry = tk.Text(home_frame, height=3, width=50)
user_input_entry.pack()

ask_button = tk.Button(home_frame, text="Ask", command=get_response, height=2, width=10)
ask_button.pack(pady=20)

root.mainloop()
