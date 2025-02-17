# 🤖 Customer Segmentation Chatbot using Gemini and ChromaDB (RAG Chatbot)  

This repository contains a project that demonstrates how to build a **Customer Segmentation Chatbot** using **Google Gemini**, **ChromaDB**, and **Sentence Transformers**. The chatbot processes customer-related queries by leveraging historical customer data stored in **ChromaDB**, embedding it into vectors, and retrieving it for **context-based question answering**.  

This implementation follows the **RAG (Retrieval-Augmented Generation) paradigm**, meaning it **retrieves relevant information** from a knowledge base and **enhances responses** by incorporating that context.  

## 🚀 Features  
✅ Uses **Google Gemini API** for natural language responses.  
✅ **Retrieval-Augmented Generation (RAG)** for more informed and accurate answers.  
✅ Stores **customer interaction history** to provide continuity in conversations.  
✅ **Gradio-based UI** for seamless user interaction.  
✅ **ChromaDB** stores vector embeddings for fast and efficient retrieval.  

---

## 📌 Table of Contents  
- [🛠️ Prerequisites](#️-prerequisites)  
- [📥 Setup and Installation](#-setup-and-installation)  
- [📂 Code Structure and Functionality](#-code-structure-and-functionality)  
- [⚙️ How the Chatbot Works](#️-how-the-chatbot-works)  
- [▶️ Running the Application](#-running-the-application)  
- [🏁 Conclusion](#-conclusion)  

---

## 🛠️ Prerequisites  
Before running the application, ensure you have:  
- **Python 3.7+**  
- **pip** installed  
- The following Python libraries:  
  ```bash
  pip install chromadb sentence-transformers google-genai gradio python-dotenv
  ```
- **Google Gemini API key** (stored securely as shown below).  

---

## 📥 Setup and Installation  

### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/your-repo/customer-segmentation-chatbot.git
cd customer-segmentation-chatbot
```

### 2️⃣ Install Dependencies  
```bash
pip install -r requirements.txt
```

### 3️⃣ Set Up Environment Variables  
- Create a `.env` file in the root directory.  
- Add your API key:  
  ```bash
  GEMINI_API_KEY=your_api_key_here
  ```
- **Important**: Add `.env` to `.gitignore` to keep it secure.  

---

## 📂 Code Structure and Functionality  

### ⚙️ Environment Setup  
- Loads API key using `dotenv`:  
  ```python
  from dotenv import load_dotenv
  load_dotenv()  
  api_key = os.getenv("GEMINI_API_KEY")
  ```
  
### 🏦 File Handling & ChromaDB Setup  
- **ChromaDB Initialization** for storing customer data:  
  ```python
  client = chromadb.Client(Settings(persist_directory="./chroma_db"))
  collection = client.get_or_create_collection("customer_segments")
  ```
- Reads and chunks **customer data** from `Trial/R1.txt` before embedding it.  

### 🧠 Embedding & Contextual Retrieval (RAG)  
- Uses **Sentence Transformers** to convert text into embeddings for retrieval:  
  ```python
  model = SentenceTransformer('all-MiniLM-L6-v2')
  embedding = model.encode(chunk).tolist()
  ```
- Chunks the text into **50-word segments** for efficient retrieval.  

### 🔍 Gemini API Integration  
- Converts the **user query** into an embedding:  
  ```python
  def get_query_embedding(query):
      return model.encode([query])[0]
  ```
- Retrieves relevant chunks and **augments responses** using session memory.  

### 🧠 Session Memory & Conversation History  
- Stores **user interactions** in a session:  
  ```python
  user_history.append({"question": query, "answer": answer})
  ```
- Ensures **continuity in conversations** using previous interactions.  

### 🎨 UI Implementation with Gradio  
- Creates a **chat UI** with buttons for restarting and ending conversations:  
  ```python
  with gr.Blocks() as chatbot_ui:
      gr.Markdown("## 🤖 Customer Segmentation Chatbot")
      restart_button = gr.Button("Restart Conversation")
      chat_display = gr.Chatbot(label="Chat History", type='messages')
      user_input = gr.Textbox(label="Type your message:")
      end_button = gr.Button("End Conversation")
  ```
  
---

## ⚙️ How the Chatbot Works  

1️⃣ **User Inputs a Query** →  
2️⃣ **ChromaDB Retrieves Relevant Context** →  
3️⃣ **Gemini Generates a Response** →  
4️⃣ **Session Memory Enhances Context** →  
5️⃣ **User Continues or Ends the Conversation**  

💡 **This RAG approach ensures every response is informed by both historical data and ongoing conversation!**  

---

## ▶️ Running the Application  
1. Clone the repository and navigate to it:  
   ```bash
   git clone https://github.com/KRV2928/RAGCHAT.git
   cd RAGCHAT
   ```
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```
3. Run the chatbot:  
   ```bash
   python chatbot.py
   ```
4. Open the **Gradio UI** in your browser and start chatting!  

---

## 🏁 Conclusion  
This **RAG-powered chatbot** efficiently handles **customer segmentation queries** by retrieving and utilizing relevant context. The **session memory** enables more **engaging and meaningful conversations**.  

🚀 **Customize and extend** the chatbot for use cases like:  
✅ **Customer support**  
✅ **Market research**  
✅ **Personalized recommendations**  

--- 
