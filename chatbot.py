import os
from tqdm.auto import tqdm
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import textwrap
from google import genai
import gradio as gr
from dotenv import load_dotenv

# Initialize ChromaDB and create collection
client = chromadb.Client(Settings(persist_directory="./chroma_db"))
collection_name = "customer_segments"

# Create or get collection
try:
    collection = client.get_collection(collection_name)
except:
    collection = client.create_collection(collection_name)

# Load the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# .. moves it one level up. Relative path from 'code' to 'input_files'
file_path = 'R1.txt'

# Check if the file exists and is accessible
if os.path.isfile(file_path):
    try:
        with open(file_path, 'r') as file:
            text = file.read()
        
        # Check if the file content is not empty
        if text.strip():  
            print("✅ File read successfully.")
        else:
            print("⚠️ File is empty.")
    
    except Exception as e:
        print(f"❌ Failed to read the file: {e}")
else:
    print(f"❌ File not found or inaccessible: {file_path}")

    # Chunk the text into smaller parts
chunk_size = 100  # Adjust based on your requirements
chunks = textwrap.wrap(text, width=chunk_size)

# Generate embeddings and store them
for idx, chunk in enumerate(chunks):
    embedding = model.encode(chunk).tolist()
    collection.add(
        documents=[chunk],
        embeddings=[embedding],
        ids=[str(idx)]
    )

print(f"Stored {len(chunks)} chunks in ChromaDB.")

collection = client.get_or_create_collection("customer_segments")

# Function to get the embedding for a query
def get_query_embedding(query):
    return model.encode([query])[0]

# Function to retrieve the most relevant chunks
def retrieve_relevant_chunks(query, top_k=10):
    query_embedding = get_query_embedding(query)
    
    # Retrieve top_k most similar chunks from ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    
    # Flatten results to ensure you're working with just the strings
    chunks = [chunk for chunk in results['documents']]

    return results['documents']

# Load the environment variables
load_dotenv()

# Set your Gemini API key and endpoint
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Initialize the Gemini client with the API key
client = genai.Client(api_key=f"{gemini_api_key}")

# Global variable to hold session memory
user_history = []

def synthesize_response_with_gemini(chunks, query):
    global user_history  # Use the global session memory

    # Ensure chunks are strings (in case there's any list or nested structure)
    chunks = [str(chunk) for chunk in chunks]

    # Combine the relevant chunks into a single string for context
    context = "\n".join(chunks)

    # Add the session history to the context (this includes the Q&A history)
    conversation_history = "\n".join([f"Q: {entry['question']}\nA: {entry['answer']}" for entry in user_history])

    # Prepare the input text for the Gemini model
    contents = (
        f"Answer the following question. Use the provided context if it's relevant; otherwise, use your general knowledge. "
        f"Do not mention context unavailability; just answer the question directly.\n"
        f"Context (USE ONLY THE AVAILABLE CONTEXT, ):\n{context}\n\nConversation History (if any):\n{conversation_history}\n\nQuestion: {query}\nAnswer:"
    )

    try:
        # Use the Gemini client to generate content
        response = client.models.generate_content(
            model="gemini-2.0-flash",  # Use the desired model
            contents=contents
        )

        # Extract and clean the response text
        answer = response.text.strip() if response.text else "Sorry, I couldn't generate a response."

        # Remove any leading text like 'Answer:' if present
        if answer.lower().startswith("answer:"):
            answer = answer[7:].strip()

        # Save the current question and answer to the session memory
        user_history.append({"question": query, "answer": answer})

        return answer

    except Exception as e:
        # Handle potential errors gracefully
        return f"Can you please rephrase the question for me?"
    

# Global variable to track input state
input_disabled = False
session_memory = []

# Function to handle chat with Gemini
def chat_with_gemini(user_input):
    global input_disabled

    # Check if the user wants to end the conversation
    if user_input.lower() == "end":
        session_memory.clear()  # Clear the memory
        input_disabled = True
        return {"role": "assistant", "content": "The conversation has been ended. Please click 'End Conversation' to exit."}

    # Retrieve relevant chunks based on the current query
    relevant_chunks = retrieve_relevant_chunks(user_input)

    # Generate response using Gemini
    answer = synthesize_response_with_gemini(relevant_chunks, user_input)

    # Add the current exchange to session memory
    session_memory.append({"question": user_input, "answer": answer})

    return {"role": "assistant", "content": answer}


# Function to handle UI shutdown gracefully
def shutdown_ui():
    return gr.update(visible=False), gr.update(visible=False), gr.HTML("<h2>Thank you for using the Customer Segmentation Chatbot! 👋</h2>")

# Function to restart the session
def restart_session():
    global session_memory, input_disabled
    session_memory.clear()
    input_disabled = False
    return "", [], gr.update(interactive=True, visible=True), gr.update(visible=True), gr.HTML("<h2>New Session Started! 🎉</h2>")

# Create Gradio interface
with gr.Blocks() as chatbot_ui:
    gr.Markdown("## 🤖 Customer Segmentation Chatbot")

    restart_button = gr.Button("Restart Conversation")
    chat_display = gr.Chatbot(label="Chat History", type='messages')
    user_input = gr.Textbox(label="Type your message:", placeholder="Ask me something about customer segments...", interactive=True)
    end_button = gr.Button("End Conversation")
    thank_you_msg = gr.HTML("", visible=True)

    def handle_user_input(user_input_text, chat):
        global input_disabled

        if input_disabled:
            return "", chat, gr.update(interactive=False), gr.update(visible=True) # Keep button visible

        response = chat_with_gemini(user_input_text)
        chat.append({"role": " ", "content": user_input_text})
        chat.append(response)

        # Disable input if the user types "end"
        if user_input_text.lower() == "end":
            return "", chat, gr.update(interactive=False, visible=False), gr.update(visible=True)

        return "", chat, gr.update(interactive=True), gr.update(visible=True)

    user_input.submit(
        handle_user_input,
        inputs=[user_input, chat_display],
        outputs=[user_input, chat_display, user_input, end_button]
    )
    end_button.click(shutdown_ui, outputs=[user_input, end_button, thank_you_msg])
    restart_button.click(restart_session, outputs=[user_input, chat_display, user_input, end_button, thank_you_msg])

chatbot_ui.launch(share=True)