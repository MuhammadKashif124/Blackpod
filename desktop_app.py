from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.core.window import Window
from kivy.graphics import Color, RoundedRectangle, Line
from kivy.uix.image import Image
from kivy.lang import Builder
from kivy.uix.scrollview import ScrollView
from kivy.uix.gridlayout import GridLayout
from kivy.animation import Animation
from kivy.clock import Clock
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from kivy.uix.dropdown import DropDown
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.popup import Popup
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.widget import Widget
from threading import Thread
from functools import partial
from user import UserManager
from chatbot import qa_system

# Add this right after the imports
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# Import and initialize ChromaConfig
from config import ChromaConfig

# Initialize ChromaConfig with EC2 setting from environment
chroma_config = ChromaConfig(use_api=os.getenv('USE_API', 'false').lower() == 'true')
VECTOR_DB_DIR = chroma_config.db_dir

class RoundedButton(Button):
    pass

class RoundedTextInput(TextInput):
    pass

Builder.load_string('''
<RoundedButton@Button>:
    background_color: 0,0,0,0
    background_normal: ''
    font_name: 'Roboto'
    color: [1, 1, 1, 1]  # White text for buttons
    canvas.before:
        Color:
            rgba: (0.25, 0.5, 1, 1) if self.state == 'normal' else (0.2, 0.4, 0.9, 1)
        RoundedRectangle:
            pos: self.pos
            size: self.size
            radius: [25,]
    canvas.after:
        Color:
            rgba: (1, 1, 1, 0.1)
        Line:
            rounded_rectangle: (self.pos[0], self.pos[1], self.size[0], self.size[1], 25)
            width: 1.2

<RoundedTextInput>:
    background_color: [0.12, 0.12, 0.17, 1]
    foreground_color: [1, 1, 1, 1]  # White text
    cursor_color: [1, 1, 1, 1]  # White cursor
    padding: [25, 15]
    font_size: '16sp'
    font_name: 'Roboto'
    hint_text_color: [0.7, 0.7, 0.7, 1]  # Lighter gray for better visibility
    selection_color: [0.25, 0.5, 1, 0.5]
    write_tab: False
    background_normal: ''
    background_active: ''
    use_bubble: False  # Disable the bubble effect
    use_handles: False  # Disable selection handles
    text_validate_unfocus: False
    canvas.before:
        Clear
        Color:
            rgba: [0.18, 0.18, 0.23, 1]
        RoundedRectangle:
            pos: self.pos
            size: self.size
            radius: [25,]
        Color:
            rgba: [0.25, 0.5, 1, 0.1]
        Line:
            rounded_rectangle: (self.pos[0], self.pos[1], self.size[0], self.size[1], 25)
            width: 1.2

<Label>:
    color: [1, 1, 1, 1]  # White text for all labels
    font_name: 'Roboto'
''')


class MessageBubble(BoxLayout):
    def __init__(self, message, is_user=False, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'horizontal'
        self.size_hint_y = None
        self.padding = [10, 5]
        self.spacing = 10
        self.height = 40  # Initial height, will be adjusted
        
        # Create message container
        msg_container = BoxLayout()
        msg_container.size_hint_x = 0.8
        msg_container.size_hint_y = None
        msg_container.padding = [15, 10]
        
        with msg_container.canvas.before:
            Color(rgba=(0.25, 0.5, 1, 1) if is_user else (0.18, 0.18, 0.23, 1))
            self.rect = RoundedRectangle(radius=[15])
        
        # Message label with white text
        msg_label = Label(
            text=message,
            color=[1, 1, 1, 1],  # Pure white text
            size_hint_y=None,
            font_size='16sp',
            font_name='Roboto',
            halign='left',
            valign='middle',
            text_size=(None, None),  # Will be set in _update_label_text_size
            markup=True  # Enable markup for better text handling
        )
        
        msg_container.bind(size=self._update_label_text_size)
        msg_container.bind(pos=self._update_rect)
        msg_container.bind(size=self._update_rect)
        
        msg_container.add_widget(msg_label)
        
        # Add spacing for message alignment
        if is_user:
            self.add_widget(BoxLayout(size_hint_x=0.2))
        self.add_widget(msg_container)
        if not is_user:
            self.add_widget(BoxLayout(size_hint_x=0.2))
        
        # Animate the bubble appearance
        self.opacity = 0
        anim = Animation(opacity=1, duration=0.3)
        anim.start(self)
    
    def _update_label_text_size(self, instance, value):
        label = instance.children[0]
        # Set text_size to enable wrapping
        label.text_size = (instance.width - 20, None)
        # Let the label calculate its size
        label.texture_update()
        # Update heights based on content
        new_height = max(label.texture_size[1], label.line_height)
        label.height = new_height
        instance.height = new_height + 20  # Add padding
        self.height = instance.height + 10  # Update bubble height
    
    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size

class ChatArea(ScrollView):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.messages = GridLayout(cols=1, spacing=10, size_hint_y=None, padding=[10, 10])
        self.messages.bind(minimum_height=self.messages.setter('height'))
        self.add_widget(self.messages)
    
    def add_message(self, text, is_user=False):
        message = MessageBubble(text, is_user=is_user)
        self.messages.add_widget(message)
        Clock.schedule_once(lambda dt: self.scroll_to(message), 0.1)

class QASystem:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=api_key,
            model="text-embedding-3-large"
        )
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.7,
            openai_api_key=api_key
        )
        self.current_db = None
        self.qa_chain = None
        self.current_vectorstore = None

    def load_database(self, db_name):
        try:
            # Construct the full database path
            full_db_path = os.path.join(VECTOR_DB_DIR, db_name)
            print(f"Full database path: {full_db_path}")
            
            if not os.path.exists(full_db_path):
                raise FileNotFoundError(f"Database path not found: {full_db_path}")
            
            self.current_vectorstore = Chroma(
                persist_directory=full_db_path,
                embedding_function=self.embeddings
            )
            
            # Create the prompt template with better context handling
            custom_template = """
            You are a knowledgeable AI assistant. Use ONLY the following context and chat history to answer the question. 
            If you cannot find the answer in the context, say so clearly.

            Context: {context}

            Chat History: {chat_history}

            Current Question: {question}

            Assistant: Let me help you with that based on the provided context.
            """
            
            PROMPT = PromptTemplate(
                input_variables=["context", "chat_history", "question"],
                template=custom_template
            )
            
            self.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.current_vectorstore.as_retriever(search_kwargs={"k": 4}),
                return_source_documents=True,
                combine_docs_chain_kwargs={"prompt": PROMPT}
            )
            
            self.current_db = db_name
            return True
        except Exception as e:
            print(f"Error loading database: {str(e)}")
            return False

    def get_available_databases(self):
        try:
            # Normalize the path to ensure consistent format
            db_path = os.path.normpath(VECTOR_DB_DIR)
            if not os.path.exists(db_path):
                print(f"Vector DB directory not found: {db_path}")
                return []
            
            databases = [d for d in os.listdir(db_path) 
                        if os.path.isdir(os.path.join(db_path, d))]
            print(f"Found databases: {databases}")
            return databases
        except Exception as e:
            print(f"Error accessing database directory: {str(e)}")
            return []

# Initialize QA system
qa_system = QASystem()

class LoginScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        layout = FloatLayout()
        
        # Title label
        title_label = Label(
            text="BlackPod AI",
            font_size='24sp',
            color=[1, 1, 1, 1],
            pos_hint={'center_x': 0.5, 'center_y': 0.85},
            size_hint=(0.8, 0.1)
        )
        
        # Username input
        self.username = TextInput(
            hint_text='Username',
            size_hint=(0.8, None),
            height=50,
            pos_hint={'center_x': 0.5, 'center_y': 0.65},
            foreground_color=[1, 1, 1, 1],
            hint_text_color=[0.7, 0.7, 0.7, 1],
            background_color=[0.12, 0.12, 0.17, 1],
            cursor_color=[1, 1, 1, 1],
            background_normal='',
            background_active='',
            multiline=False,
            write_tab=False,
            padding=[25, 15],
            font_name='Roboto',
            font_size='16sp'
        )
        
        # Password input
        self.password = TextInput(
            hint_text='Password',
            password=True,
            size_hint=(0.8, None),
            height=50,
            pos_hint={'center_x': 0.5, 'center_y': 0.55},
            foreground_color=[1, 1, 1, 1],
            hint_text_color=[0.7, 0.7, 0.7, 1],
            background_color=[0.12, 0.12, 0.17, 1],
            cursor_color=[1, 1, 1, 1],
            background_normal='',
            background_active='',
            multiline=False,
            write_tab=False,
            padding=[25, 15],
            font_name='Roboto',
            font_size='16sp'
        )

        # Organization Key input
        self.org_key = TextInput(
            hint_text='Organization Key',
            size_hint=(0.8, None),
            height=50,
            pos_hint={'center_x': 0.5, 'center_y': 0.45},
            foreground_color=[1, 1, 1, 1],
            hint_text_color=[0.7, 0.7, 0.7, 1],
            background_color=[0.12, 0.12, 0.17, 1],
            cursor_color=[1, 1, 1, 1],
            background_normal='',
            background_active='',
            multiline=False,
            write_tab=False,
            padding=[25, 15],
            font_name='Roboto',
            font_size='16sp'
        )
        
        # Login button
        login_btn = RoundedButton(
            text='Login',
            size_hint=(0.8, None),
            height=50,
            pos_hint={'center_x': 0.5, 'center_y': 0.35},
            color=[1, 1, 1, 1]
        )
        login_btn.bind(on_release=self.attempt_login)
        
        # Add widgets to layout in correct order
        layout.add_widget(title_label)
        layout.add_widget(self.username)
        layout.add_widget(self.password)
        layout.add_widget(self.org_key)
        layout.add_widget(login_btn)
        
        # Add layout to screen
        self.add_widget(layout)
    
    def attempt_login(self, instance):
        username = self.username.text.strip()
        password = self.password.text.strip()
        org_key = self.org_key.text.strip()
        
        # Basic validation
        if not all([username, password, org_key]):
            self.show_error("Please fill in all fields")
            return
        
        # Use the actual authentication logic
        user_manager = UserManager()
        result = user_manager.login(username, password, org_key)
        
        if "Login successful" in result:
            # Store the organization key and try to load the corresponding database
            App.get_running_app().org_key = org_key
            
            # First check if the database exists
            if os.path.exists(os.path.join(VECTOR_DB_DIR, org_key)):
                if qa_system.load_database(org_key):
                    self.manager.current = 'chat'
                else:
                    self.show_error("Error loading database. Please try again.")
            else:
                self.show_error(f"No database found for organization key: {org_key}")
        else:
            self.show_error(result)
    
    def show_error(self, message):
        content = Label(
            text=message,
            color=[1, 0, 0, 1],  # Red for errors
            font_size='16sp'
        )
        popup = Popup(
            title='Error',
            content=content,
            size_hint=(0.8, 0.2),
            title_color=[1, 1, 1, 1],  # White title
            title_size='18sp',
            background_color=[0.18, 0.18, 0.23, 1]  # Dark background
        )
        popup.open()

class DatabaseSelectScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Main layout with padding
        layout = BoxLayout(orientation='vertical', padding=[25, 40, 25, 25], spacing=30)
        
        # Header area with title and description
        header_layout = BoxLayout(orientation='vertical', spacing=10, size_hint_y=None, height=100)
        
        # Title with larger font
        title = Label(
            text="Select Database",
            font_size='28sp',
            color=[1, 1, 1, 1],
            bold=True,
            size_hint_y=None,
            height=50
        )
        
        # Description text
        description = Label(
            text="Choose a database to start chatting",
            font_size='16sp',
            color=[0.7, 0.7, 0.7, 1],
            size_hint_y=None,
            height=30
        )
        
        header_layout.add_widget(title)
        header_layout.add_widget(description)
        
        # Add header to main layout
        layout.add_widget(header_layout)
        
        # Spacer
        layout.add_widget(Widget(size_hint_y=0.1))
        
        # Database selection area
        selection_layout = BoxLayout(orientation='vertical', spacing=15, size_hint_y=None, height=150)
        
        # Main button that will show the dropdown
        self.main_button = RoundedButton(
            text='Select Database',
            size_hint=(1, None),
            height=50,
            background_color=[0.25, 0.5, 1, 1]  # Blue color
        )
        
        # Create the dropdown with custom styling
        self.dropdown = DropDown(
            auto_width=False,
            size_hint_x=1,  # Make dropdown same width as button
        )
        
        # Add canvas instructions to style the dropdown background
        with self.dropdown.canvas.before:
            Color(rgba=(0.12, 0.12, 0.17, 0.95))  # Dark background with slight transparency
            self.dropdown.rect = RoundedRectangle(size=self.dropdown.size, pos=self.dropdown.pos, radius=[15])
        
        # Bind dropdown size and position updates
        self.dropdown.bind(size=self._update_dropdown_rect, pos=self._update_dropdown_rect)
        
        # Status label with white text
        self.status_label = Label(
            text="No database selected",
            color=[0.7, 0.7, 0.7, 1],  # Light gray for status
            size_hint_y=None,
            height=30,
            font_size='14sp'
        )
        
        selection_layout.add_widget(self.main_button)
        selection_layout.add_widget(self.status_label)
        
        # Add selection area to main layout
        layout.add_widget(selection_layout)
        
        # Spacer
        layout.add_widget(Widget(size_hint_y=0.2))
        
        # Continue button at the bottom
        continue_btn = RoundedButton(
            text='Continue',
            size_hint=(1, None),
            height=50,
            background_color=[0.25, 0.5, 1, 1]  # Blue color
        )
        continue_btn.bind(on_release=self.proceed_to_chat)
        
        # Add continue button to main layout
        layout.add_widget(continue_btn)
        
        # Load databases and bind dropdown
        self.load_databases()
        self.main_button.bind(on_release=self.dropdown.open)
        
        # Add the main layout to the screen
        self.add_widget(layout)
    
    def _update_dropdown_rect(self, instance, value):
        """Update the dropdown background rectangle"""
        instance.rect.size = instance.size
        instance.rect.pos = instance.pos
    
    def load_databases(self):
        # Get the organization key from the app instance
        org_key = App.get_running_app().org_key
        
        # Get all databases
        databases = qa_system.get_available_databases()
        
        # Handle case where org_key is not set
        if not org_key:
            self.status_label.text = "Please log in first"
            return
        
        # Filter databases based on organization key
        filtered_databases = [db for db in databases if db.startswith(org_key)]
        
        if filtered_databases:
            for db in filtered_databases:
                # Create a custom styled button for each database
                btn = Button(
                    text=db,
                    size_hint_y=None,
                    height=45,
                    background_color=[0, 0, 0, 0],  # Transparent background
                    background_normal='',
                    background_down='',
                    color=[1, 1, 1, 1],  # White text
                    font_size='16sp'
                )
                
                # Add hover effect
                btn.bind(
                    on_release=lambda btn: self.select_database(btn.text)
                )
                self.dropdown.add_widget(btn)
            self.status_label.text = f"Found {len(filtered_databases)} databases for your organization"
        else:
            self.status_label.text = "No databases available for your organization"
    
    def _update_btn_rect(self, instance, value):
        """Update the button background rectangle"""
        instance.rect.size = instance.size
        instance.rect.pos = instance.pos
    
    def _update_btn_state(self, instance, value):
        """Update button appearance based on state"""
        if value == 'down':
            instance.bg_color.rgba = (0.25, 0.5, 1, 1)  # Blue when pressed
        else:
            instance.bg_color.rgba = (0.18, 0.18, 0.23, 1)  # Default dark background
    
    def select_database(self, db_name):
        try:
            selected_db = os.path.normpath(os.path.join(VECTOR_DB_DIR, db_name))
            print(f"Attempting to load database from: {selected_db}")
            
            if qa_system.load_database(selected_db):
                self.selected_db = selected_db
                self.main_button.text = db_name
                self.status_label.text = f"Selected: {db_name}"
                self.status_label.color = [0.3, 0.8, 0.3, 1]  # Green for success
                print(f"Successfully loaded database: {db_name}")
            else:
                self.status_label.text = f"Failed to load: {db_name}"
                self.status_label.color = [0.8, 0.3, 0.3, 1]  # Red for failure
                print(f"Failed to load database: {db_name}")
            self.dropdown.dismiss()
        except Exception as e:
            self.status_label.text = f"Error: {str(e)}"
            self.status_label.color = [0.8, 0.3, 0.3, 1]  # Red for error
            print(f"Error selecting database: {str(e)}")
            self.dropdown.dismiss()
    
    def proceed_to_chat(self, instance):
        if hasattr(self, 'selected_db'):
            self.manager.current = 'chat'
        else:
            self.show_error("Please select a database first")
    
    def show_error(self, message):
        content = BoxLayout(orientation='vertical', padding=[20, 10])
        content.add_widget(Label(
            text=message,
            color=[1, 0.3, 0.3, 1],  # Light red for errors
            font_size='16sp'
        ))
        popup = Popup(
            title='Error',
            content=content,
            size_hint=(0.8, 0.3),
            title_color=[1, 1, 1, 1],  # White title
            title_size='18sp',
            background_color=[0.18, 0.18, 0.23, 1]  # Dark background
        )
        popup.open()

class ChatScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layout = BoxLayout(orientation='vertical', padding=25, spacing=20)
        
        # Chat area
        self.chat_area = ChatArea(size_hint=(1, 1))
        self.layout.add_widget(self.chat_area)
        
        # Input area with explicit style overrides - using same config as login fields
        self.user_input = TextInput(
            hint_text='Type your request...',
            size_hint_y=None,
            height=50,
            foreground_color=[1, 1, 1, 1],  # White text
            hint_text_color=[0.7, 0.7, 0.7, 1],
            background_color=[0.12, 0.12, 0.17, 1],
            cursor_color=[1, 1, 1, 1],
            background_normal='',
            background_active='',
            multiline=False,
            write_tab=False,
            padding=[25, 15],
            font_name='Roboto',
            font_size='16sp',
            use_bubble=False,  # Disable the bubble effect
            use_handles=False  # Disable selection handles
        )
        self.user_input.bind(on_text_validate=self.on_text_submit)
        self.layout.add_widget(self.user_input)
        
        # Initialize chat history as list of tuples (question, answer)
        self.chat_history = []
        
        # Add a loading indicator
        self.loading_label = Label(
            text="Processing...",
            color=[0.7, 0.7, 0.7, 1],
            size_hint_y=None,
            height=30,
            opacity=0
        )
        self.layout.add_widget(self.loading_label)
        
        self.add_widget(self.layout)
    
    def show_loading(self, show=True):
        self.loading_label.opacity = 1 if show else 0
        self.user_input.disabled = show
    
    def process_chat_in_thread(self, user_message):
        try:
            # Format chat history for the chain
            formatted_history = [(q, a) for q, a in self.chat_history]
            
            # Get response from QA system with chat history
            response = qa_system.qa_chain.invoke({
                "question": user_message,
                "chat_history": formatted_history
            })
            
            # Extract answer and source documents
            bot_message = response.get('answer', "I couldn't generate a response.")
                
            # Add source attribution if available
            source_docs = response.get('source_documents', [])
            if source_docs:
                sources = "\n\nSources:"
                for i, doc in enumerate(source_docs, 1):
                    if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                        sources += f"\n{i}. {doc.metadata['source']}"
            
            # Update chat history with the new exchange
            self.chat_history.append((user_message, bot_message))
            
            # Limit chat history to last 10 exchanges to prevent context overflow
            if len(self.chat_history) > 10:
                self.chat_history = self.chat_history[-10:]
            
            # Schedule the UI update on the main thread
            Clock.schedule_once(lambda dt: self.update_chat_ui(bot_message), 0)
            
        except Exception as e:
            bot_message = "I apologize, but I encountered an error processing your request. Please try again."
            print(f"Error in chat: {str(e)}")
            Clock.schedule_once(lambda dt: self.update_chat_ui(bot_message), 0)
        
        finally:
            # Hide loading indicator
            Clock.schedule_once(lambda dt: self.show_loading(False), 0)
    
    def update_chat_ui(self, bot_message):
        self.chat_area.add_message(bot_message, is_user=False)
    
    def on_text_submit(self, instance):
        user_message = instance.text.strip()
        if user_message:
            # Clear input
            instance.text = ''
            
            # Add user message to chat
            self.chat_area.add_message(user_message, is_user=True)
            
            # Show loading indicator
            self.show_loading(True)
            
            if not qa_system.current_db:
                self.chat_area.add_message("Please select a database first!", is_user=False)
                self.show_loading(False)
                return
            
            # Process chat in background thread
            Thread(
                target=self.process_chat_in_thread,
                args=(user_message,),
                daemon=True
            ).start()

class ChatBotApp(App):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.org_key = None  # Initialize org_key
        self.sm = ScreenManager()

    def build(self):
        # Add screens to the screen manager
        self.sm.add_widget(LoginScreen(name='login'))
        self.sm.add_widget(DatabaseSelectScreen(name='database_select'))
        self.sm.add_widget(ChatScreen(name='chat'))
        return self.sm

    # Update this when loading databases
    def update_status(self, text):
        self.status_label.text = text

if __name__ == '__main__':
    ChatBotApp().run()