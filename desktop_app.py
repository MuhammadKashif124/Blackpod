import logging
logging.basicConfig(filename='app_launch.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Application starting up...")

import os
import sys
import traceback  # Add traceback for better error logging

# Conditional import of fcntl - only for non-Windows systems
if os.name != 'nt':  # 'nt' is the os.name for Windows
    import fcntl
    has_fcntl = True
else:
    has_fcntl = False
    fcntl = None # To avoid NameError later
    # Import Windows-specific modules for single instance check
    import ctypes
    import msvcrt
    from ctypes import wintypes

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.textinput import TextInput
from kivy.core.window import Window
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.lang import Builder
from kivy.uix.widget import Widget
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
import subprocess
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
import logging
import json
from kivy.clock import Clock
from functools import partial
import threading
from kivy.animation import Animation  # Add this import for Animation
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.relativelayout import RelativeLayout

# Configure logging - update to include more detailed error logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('desktop_app.log'),
        logging.StreamHandler()
    ]
)

# Add a function to handle uncaught exceptions
def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        # Don't log keyboard interrupt
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    
    logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    # Display error message to user
    from kivy.app import App
    from kivy.uix.popup import Popup
    from kivy.uix.label import Label
    
    if App.get_running_app():
        error_msg = f"An error occurred: {exc_value}"
        popup = Popup(title='Error',
                    content=Label(text=error_msg),
                    size_hint=(0.8, 0.4))
        popup.open()

# Set the exception handler
sys.excepthook = handle_exception

# Import components
try:
    from components.login_screen import LoginScreen
    from components.database_select_screen import DatabaseSelectScreen
    from components.chat_area import ChatArea
    from components.rounded_button import RoundedButton
except ImportError as e:
    logging.error(f"Error importing components: {e}")
    raise

# Load environment variables
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# Base directory setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Pinecone setup
pinecone_api_key = os.getenv('PINECONE_API_KEY')
pinecone_environment = os.getenv('PINECONE_ENVIRONMENT')
pinecone_index_name = "test"
if not pinecone_api_key:
    raise ValueError("PINECONE_API_KEY environment variable is not set")
if not pinecone_environment:
    raise ValueError("PINECONE_ENVIRONMENT environment variable is not set")

# MongoDB setup
mongo_uri = os.getenv('MONGODB_URI')
mongo_db_name = os.getenv('DATABASE_NAME')
mongo_collection_name = os.getenv('COLLECTION_NAME')
if not mongo_uri:
    raise ValueError("MONGODB_URI environment variable is not set")
if not mongo_db_name:
    raise ValueError("DATABASE_NAME environment variable is not set")
if not mongo_collection_name:
    raise ValueError("COLLECTION_NAME environment variable is not set")

# Load Kivy styles
Builder.load_string('''
<RoundedTextInput@TextInput>:
    background_color: [0.18, 0.18, 0.23, 1]  # Same as AI message bubble
    foreground_color: [1, 1, 1, 1]  # White text
    cursor_color: [1, 1, 1, 1]  # White cursor
    padding: [25, 15]
    font_size: '16sp'
    font_name: 'Roboto'
    hint_text_color: [0.7, 0.7, 0.7, 1]
    selection_color: [0.25, 0.5, 1, 0.5]
    write_tab: False
    multiline: False
    background_normal: ''
    background_active: ''
    background_disabled: ''
    background_disabled_normal: ''
    use_bubble: False
    use_handles: False
    text_validate_unfocus: False
    halign: 'left'
    valign: 'middle'
    canvas.before:
        Clear
        Color:
            rgba: [0.18, 0.18, 0.23, 1]  # Same as AI message bubble
        RoundedRectangle:
            pos: self.pos
            size: self.size
            radius: [15]  # Same radius as message bubbles
        Color:
            rgba: [0.25, 0.5, 1, 0.1]  # Subtle blue border
        Line:
            rounded_rectangle: (self.pos[0], self.pos[1], self.size[0], self.size[1], 15)
            width: 1.2

<Label>:
    color: [1, 1, 1, 1]  # White text for all labels
    font_name: 'Roboto'
''')

# Comment out or remove this entire class
# class RoundedTextInput(TextInput):
#     def __init__(self, **kwargs):
#         # ... existing code ...

class QASystem:
    def __init__(self):
        self.initialize()
        
    def initialize(self):
        try:
            # Initialize OpenAI components with error handling
            self.embeddings = OpenAIEmbeddings(
                openai_api_key=api_key,
                model="text-embedding-3-large"
            )
            self.llm = ChatOpenAI(
                model="gpt-4o-mini",  # Updated to latest model
                temperature=0.7,
                openai_api_key=api_key
            )
            
            # Force reload of environment variables to get the latest database selection
            from dotenv import load_dotenv
            load_dotenv(override=True)
            
            # Get the active Pinecone index URL from environment
            # First check if a database was selected in the UI
            pinecone_index_url = os.getenv('ACTIVE_PINECONE_INDEX_URL')
            
            # Fall back to the main PINECONE_INDEX_URL if no selection was made
            if not pinecone_index_url:
                pinecone_index_url = os.getenv('PINECONE_INDEX_URL')
                
            if not pinecone_index_url:
                raise ValueError("No Pinecone index URL available. Please select a database.")
            
            logging.info(f"Using Pinecone index URL: {pinecone_index_url}")
            
            # Completely new Pinecone connection for each initialization
            if hasattr(self, 'pc') and self.pc is not None:
                try:
                    # Try to clean up old connection
                    del self.pc
                except:
                    pass
                    
            self.pc = Pinecone(api_key=pinecone_api_key)
            self.index = self.pc.Index(
                name=pinecone_index_name,
                host=pinecone_index_url.split('https://')[1]  # Extract host from URL
            )
            
            # Configure vector store with optimized settings - with no caching
            self.vectorstore = PineconeVectorStore(
                index=self.index,
                embedding=self.embeddings,
                text_key="extra_key_data",  # Specify text key for metadata
                namespace=""  # Use default namespace
            )
            
            # Initialize chat history
            self.chat_history = []
            
            logging.info(f"Successfully initialized QA System with Pinecone URL: {pinecone_index_url}")
            self.current_db_url = pinecone_index_url
        except Exception as e:
            logging.error(f"Error initializing QA System: {str(e)}")
            raise
        
        # Generic prompt template that can adapt to different contexts
        template = """
        You are a versatile AI assistant that adapts to different roles based on the context provided. You provide helpful, accurate, and relevant information based on the documents and context available to you.

        Here is the relevant information from the knowledge base:
        {context}

        Previous conversation:
        {chat_history}

        Current Question: {question}

        Instructions:
        1. Analyze the context to understand what domain or topic the user is asking about
        2. Provide accurate information based on the context provided
        3. Explain complex concepts in clear, easy-to-understand language
        4. If the context suggests a specific role (e.g., mortgage advisor, tech support, etc.), adopt that role appropriately
        5. Maintain a helpful, professional, and informative tone
        6. When appropriate, ask clarifying questions to provide better guidance
        7. If you don't have enough information to provide a specific answer, offer general guidance based on the available context
        8. Stay focused on the user's question and the relevant context
        """
        
        QA_CHAIN_PROMPT = PromptTemplate(
            input_variables=["context", "chat_history", "question"],
            template=template
        )
        
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 4}),
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT}
        )
        logging.info("QA System initialized with OpenAI models and Pinecone")

    def reinitialize(self):
        """Force a complete reinitialization of the QA system"""
        logging.info("Reinitializing QA System with fresh database connection")
        self.initialize()
        return self

    def get_response(self, query, chat_history):
        try:
            # Log which database URL we're using for this query
            pinecone_index_url = os.getenv('ACTIVE_PINECONE_INDEX_URL') or os.getenv('PINECONE_INDEX_URL')
            logging.info(f"Processing query with database URL: {pinecone_index_url}")
            
            # Get relevant documents with search
            docs = self.vectorstore.similarity_search_with_score(
                query,
                k=4  # Number of documents to retrieve
            )
            
            if not docs:
                return "I couldn't find any relevant information to answer your question."
            
            # Log documents retrieved
            logging.info(f"Retrieved {len(docs)} documents from database")
            for i, (doc, score) in enumerate(docs):
                doc_id = doc.metadata.get('id', 'N/A')
                doc_source = doc.metadata.get('source', 'Unknown')
                logging.info(f"Doc {i+1}: ID={doc_id}, Source={doc_source}, Score={score}, Content preview: {doc.page_content[:50]}...")
            
            doc_contents = []
            for doc, score in docs:
                # Enhanced metadata processing
                doc_info = {
                    "id": doc.metadata.get('id', 'N/A'),
                    "content": doc.page_content,
                    "metadata": {
                        "source": doc.metadata.get('source', 'Unknown'),
                        "document_type": doc.metadata.get('document_type', 'Unknown'),
                        "created_at": doc.metadata.get('created_at', 'Unknown')
                    },
                    "relevance_score": float(score)
                }
                doc_contents.append(doc_info)
            
            # Sort documents by relevance score
            doc_contents.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            # Format context with enhanced structure
            formatted_context = "\n\n".join([
                f"Document {i+1} (Relevance: {doc['relevance_score']:.2f}):\n" +
                f"Content: {doc['content']}\n" +
                f"Source: {doc['metadata']['source']}\n" +
                f"Type: {doc['metadata']['document_type']}"
                for i, doc in enumerate(doc_contents)
            ])
            
            # Format chat history with clear separation
            formatted_chat_history = "\n---\n".join([
                f"User: {q}\nAssistant: {a}" 
                for q, a in chat_history
            ])
            
            # Get response with enhanced context handling
            response = self.qa_chain.invoke({
                "question": query,
                "chat_history": chat_history,
                "context": formatted_context
            })
            
            answer = response.get('answer')
            if not answer:
                return "I apologize, but I couldn't generate a proper response based on the available information."
            
            logging.info(f"Successfully generated response for query: {query}")
            return answer
            
        except Exception as e:
            logging.error(f"Error in get_response: {str(e)}", exc_info=True)
            return f"I apologize, but I encountered an error while processing your request. Please try again or rephrase your question."

# Update ChatScreen class
class ChatScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Set up the screen
        self.name = 'chat'
        self.layout = BoxLayout(orientation='vertical')
        self.add_widget(self.layout)
        
        # Updated back button configuration
        back_button = RoundedButton(
            text='Back',  # Changed from arrow symbol
            size_hint=(0.15, 1),  # Match send button proportions
            background_color=[0.25, 0.5, 1, 1],  # Same blue as send button
            pos_hint={'left': 0.05, 'y': 0.05}  # Positioned at top left
        )
        back_button.bind(on_release=self.go_back)
        
        # Create a relative layout for the back button
        top_bar = RelativeLayout(size_hint=(1, 0.1))
        top_bar.add_widget(back_button)
        self.layout.add_widget(top_bar)
        
        # Initialize chat components
        self.chat_area = ChatArea(size_hint=(1, 0.85))  # Reduced to give more space to input
        
        # Create input area with proper styling
        self.input_area = BoxLayout(
            size_hint=(1, 0.15),  # Increased height 
            spacing=10,
            padding=[15, 10, 15, 10],  # Left, top, right, bottom padding
            orientation='horizontal',
            height=70  # Set a minimum height
        )
        
        # Message input - updated to match login screen style
        self.message_input = TextInput(
            hint_text='Type your message...',
            multiline=False,
            size_hint=(0.85, 1),
            height=50,
            padding=[10, 15],  # Matches login field padding
            background_color=(0.12, 0.12, 0.17, 1),  # Same dark blue
            foreground_color=(1, 1, 1, 1),  # White text
            cursor_color=(1, 1, 1, 1),  # White cursor
            font_size='16sp',
            cursor_width=2,
            background_normal='',  # Remove default background
            background_active='',  # Remove active state background
            write_tab=False  # Prevent tab characters
        )
        self.message_input.bind(on_text_validate=self.send_message)
        
        # Send button with matching style
        self.send_button = RoundedButton(
            text='Send',
            size_hint=(0.15, 1),
            background_color=[0.25, 0.5, 1, 1]  # Blue color for send button
        )
        self.send_button.bind(on_release=self.send_message)
        
        # Add components to layout with proper spacing
        input_container = BoxLayout(
            size_hint=(1, 1),
            spacing=15,
            padding=[10, 10],
            height=60,  # Set a minimum height
            minimum_height=60  # Ensure minimum height is respected
        )
        input_container.add_widget(self.message_input)
        input_container.add_widget(self.send_button)
        
        self.input_area.add_widget(input_container)
        self.layout.add_widget(self.chat_area)
        self.layout.add_widget(self.input_area)
        
        # Initialize chat history and lock
        self.chat_history = []
        self._lock = threading.Lock()
        
        # Bind to on_enter event to focus the text input when the screen is shown
        self.bind(on_enter=self._on_enter)
        
    def _on_enter(self, instance):
        # Focus the text input when the screen is shown
        Clock.schedule_once(lambda dt: self._focus_text_input(), 0.5)
        
    def _focus_text_input(self):
        # Focus the text input
        self.message_input.focus = True

    def go_back(self, instance):
        """Navigate back to the database select screen"""
        self.manager.current = 'database_select'
        # Clear chat history when going back
        self.chat_history = []
        self.chat_area.messages.clear_widgets()

    def send_message(self, instance):
        user_message = self.message_input.text.strip()
        if not user_message:
            return

        # Immediately add user message and clear input
        self.chat_area.add_message(user_message, is_user=True)
        self.message_input.text = ''
        
        def get_ai_response():
            # Show typing indicator only when starting to process
            def show_typing(dt):
                self.typing_label = self.chat_area.add_message("AI is thinking...", is_user=False)
            Clock.schedule_once(show_typing)
            
            try:
                # Get the app instance to access the QASystem
                app = App.get_running_app()
                
                # Get current active database URL
                active_url = os.getenv('ACTIVE_PINECONE_INDEX_URL') or os.getenv('PINECONE_INDEX_URL')
                
                # Check if QA system exists - if not, create it
                if not hasattr(app, 'qa_system') or app.qa_system is None:
                    # Create a new QA system
                    logging.info("Creating new QA system instance")
                    try:
                        app.qa_system = QASystem()
                        logging.info(f"Created new QA system with URL: {app.qa_system.current_db_url}")
                    except Exception as e:
                        logging.error(f"Error creating QA system: {str(e)}")
                        raise
                # Check if we need to reinitialize due to database change
                elif hasattr(app.qa_system, 'current_db_url') and app.qa_system.current_db_url != active_url:
                    logging.info(f"Database URL changed from {app.qa_system.current_db_url} to {active_url}, reinitializing")
                    app.qa_system.reinitialize()
                
                # Log which database we're using
                logging.info(f"Getting response using database URL: {active_url}")
                
                # Get AI response in background thread
                response = app.qa_system.get_response(user_message, self.chat_history)
                
                # Schedule UI update on main thread
                def update_ui(dt):
                    with self._lock:
                        # First remove typing indicator
                        if hasattr(self, 'typing_label') and self.typing_label in self.chat_area.messages.children:
                            self.chat_area.messages.remove_widget(self.typing_label)
                            self.chat_area.messages.do_layout()
                            delattr(self, 'typing_label')
                        
                        # Add the AI response after a small delay to ensure clean transition
                        def add_response(dt):
                            self.chat_area.add_message(response, is_user=False)
                            self.chat_history.append((user_message, response))
                        Clock.schedule_once(add_response, 0.1)
                
                Clock.schedule_once(update_ui)
                
            except Exception as e:
                def show_error(dt):
                    with self._lock:
                        # Remove typing indicator if it exists
                        if hasattr(self, 'typing_label') and self.typing_label in self.chat_area.messages.children:
                            self.chat_area.messages.remove_widget(self.typing_label)
                            self.chat_area.messages.do_layout()
                            delattr(self, 'typing_label')
                        error_message = f"Error: {str(e)}"
                        self.chat_area.add_message(error_message, is_user=False)
                        logging.error(f"Error in send_message: {str(e)}")
                
                Clock.schedule_once(show_error)

        # Start processing in a background thread
        threading.Thread(target=get_ai_response, daemon=True).start()

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

class BlackPodAIApp(App):
    def build(self):
        try:
            # Set window properties
            Window.size = (400, 700)
            Window.clearcolor = (0.08, 0.08, 0.12, 1)  # Dark background
            
            # Create QA system for chat functionality
            try:
                self.qa_system = QASystem()
                logging.info("QA System initialized successfully")
            except Exception as e:
                logging.error(f"Error initializing QA System: {e}")
                logging.error(traceback.format_exc())
                # Create a placeholder - will be initialized when database is selected
                self.qa_system = None
            
            # Create screen manager
            self.sm = ScreenManager()
            
            # Add screens with error handling
            try:
                login_screen = LoginScreen(name='login')
                self.sm.add_widget(login_screen)
                
                database_select = DatabaseSelectScreen(name='database_select')
                self.sm.add_widget(database_select)
                
                chat_screen = ChatScreen(name='chat')
                self.sm.add_widget(chat_screen)
                
                self.sm.current = 'login'
                logging.info("All screens added successfully")
            except Exception as e:
                logging.error(f"Error adding screens: {e}")
                logging.error(traceback.format_exc())
                raise
            
            # Start FastAPI server with better error handling
            try:
                api_script = resource_path("api_server.py")
                logging.info(f"Starting API server from: {api_script}")
                self.api_process = subprocess.Popen(
                    [sys.executable, api_script],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                logging.info("API server started successfully")
            except Exception as e:
                logging.error(f"Error starting API server: {e}")
                logging.error(traceback.format_exc())
                # Continue without API server
                self.api_process = None
            
            return self.sm
        except Exception as e:
            logging.error(f"Error in build method: {e}")
            logging.error(traceback.format_exc())
            raise

    def on_stop(self):
        try:
            if hasattr(self, 'api_process') and self.api_process:
                logging.info("Terminating API process")
                self.api_process.terminate()
        except Exception as e:
            logging.error(f"Error in on_stop: {e}")

# --- Single Instance Lock --- (Cross-platform implementation)
lock_file_path = os.path.join(os.environ.get("TEMP", "/tmp"), "BlackPodAI.lock")
lock_file = None
mutex = None

def is_already_running():
    global lock_file, mutex
    
    # Windows implementation using mutex
    if os.name == 'nt':
        mutex_name = "Global\\BlackPodAI_SingleInstance"
        try:
            logging.info(f"Creating Windows mutex: {mutex_name}")
            mutex = ctypes.windll.kernel32.CreateMutexW(None, False, mutex_name)
            last_error = ctypes.windll.kernel32.GetLastError()
            if last_error == 183:  # ERROR_ALREADY_EXISTS
                logging.warning("Windows mutex already exists. Application is already running.")
                return True
            logging.info("Windows mutex created successfully.")
            return False
        except Exception as e:
            logging.error(f"Exception during Windows mutex creation: {e}")
            return False  # Allow running on error to prevent lockout
    
    # Unix implementation using fcntl
    else:
        logging.info(f"Attempting to create lock file at: {lock_file_path}")
        try:
            lock_file = open(lock_file_path, "w")
            logging.info(f"Lock file opened successfully: {lock_file_path}")
            fcntl.lockf(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            logging.info("Lock acquired successfully.")
            return False  # No lock, so not already running
        except BlockingIOError:
            logging.warning("BlockingIOError: Lock already exists. Application is already running.")
            return True   # Lock exists, already running
        except Exception as e:
            logging.error(f"Exception during lock attempt: {e}")
            return False  # Allow running on error to prevent lockout

def cleanup_locks():
    global lock_file, mutex
    if os.name == 'nt' and mutex:
        logging.info("Closing Windows mutex")
        ctypes.windll.kernel32.CloseHandle(mutex)
    elif lock_file:
        logging.info(f"Closing and removing lock file: {lock_file_path}")
        try:
            fcntl.lockf(lock_file, fcntl.LOCK_UN)
            lock_file.close()
            os.unlink(lock_file_path)
        except Exception as e:
            logging.error(f"Error cleaning up lock file: {e}")

if __name__ == '__main__':
    # Only check for running instances when launched directly, not when imported
    logging.info("Checking if application is already running...")
    if is_already_running():
        logging.warning("Another instance of BlackPodAI is already running. Exiting.")
        print("Another instance of BlackPodAI is already running. Exiting.")
        sys.exit(0)
    
    try:
        BlackPodAIApp().run()
    finally:
        cleanup_locks()
else:
    # This code runs when the module is imported by another module
    logging.info("desktop_app.py imported as a module")