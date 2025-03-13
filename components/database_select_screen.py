from kivy.uix.screenmanager import Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.widget import Widget
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.dropdown import DropDown
from kivy.uix.popup import Popup
from kivy.graphics import Color, RoundedRectangle
from .rounded_button import RoundedButton
import os
from dotenv import load_dotenv

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

        # Proceed to Chat button
        proceed_btn = RoundedButton(
            text='Proceed to Chat',
            size_hint=(1, None),
            height=50,
            background_color=[0.25, 0.5, 1, 1]  # Blue color
        )
        proceed_btn.bind(on_release=self.proceed_to_chat)

        # Add the Proceed button to the layout
        layout.add_widget(proceed_btn)

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
        # Find all Pinecone index URLs from environment variables
        load_dotenv()  # Make sure environment variables are loaded
        
        databases = []
        # Look for all environment variables that contain Pinecone index URLs
        for key, value in os.environ.items():
            if key.startswith('PINECONE_INDEX_URL'):
                # Extract the database name from the URL
                if '://' in value:
                    url_parts = value.split('://')
                    if len(url_parts) > 1:
                        host = url_parts[1].split('.')[0]  # Extract the first part of the hostname
                        db_entry = {
                            'name': f"{host} ({key.replace('PINECONE_INDEX_URL', 'DB')})",
                            'url': value,
                            'env_key': key
                        }
                        databases.append(db_entry)
        
        if not databases:
            # Fallback to the default "test" database if no URLs found
            databases = [{'name': 'test', 'url': '', 'env_key': 'PINECONE_INDEX_URL'}]
        
        if databases:
            for db in databases:
                # Create a custom styled button for each database
                btn = Button(
                    text=db['name'],
                    size_hint_y=None,
                    height=45,
                    background_color=[0, 0, 0, 0],  # Transparent background
                    background_normal='',
                    background_down='',
                    color=[1, 1, 1, 1],  # White text
                    font_name='Roboto',
                    font_size='16sp'
                )
                
                # Store the full database info in the button
                btn.db_info = db
                
                # Add hover effect using canvas
                with btn.canvas.before:
                    btn.bg_color = Color(rgba=(0.18, 0.18, 0.23, 1))  # Store color instruction
                    btn.rect = RoundedRectangle(size=btn.size, pos=btn.pos, radius=[10])

                # Bind hover effects
                btn.bind(
                    size=self._update_btn_rect,
                    pos=self._update_btn_rect,
                    state=self._update_btn_state
                )

                # Bind the selection
                btn.bind(on_release=lambda btn=btn: self.select_database(btn.db_info))
                self.dropdown.add_widget(btn)
        else:
            btn = Button(
                text="No databases found",
                size_hint_y=None,
                height=45,
                disabled=True,
                background_color=[0.18, 0.18, 0.23, 1],
                color=[0.7, 0.7, 0.7, 1]  # Light gray text for disabled state
            )
            self.dropdown.add_widget(btn)

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

    def select_database(self, db_info):
        try:
            print(f"Attempting to load database: {db_info['name']}")
            
            # Store the full database info
            self.selected_db = db_info
            
            # Update the UI
            self.main_button.text = db_info['name']
            self.status_label.text = f"Selected: {db_info['name']}"
            self.status_label.color = [0.3, 0.8, 0.3, 1]  # Green for success
            
            # Store the selected URL in the environment for the application to use
            os.environ['ACTIVE_PINECONE_INDEX_URL'] = db_info['url']
            os.environ['ACTIVE_PINECONE_ENV_KEY'] = db_info['env_key']
            
            print(f"Successfully selected database: {db_info['name']}")
            self.dropdown.dismiss()
        except Exception as e:
            self.status_label.text = f"Error: {str(e)}"
            self.status_label.color = [0.8, 0.3, 0.3, 1]  # Red for error
            print(f"Error selecting database: {str(e)}")
            self.dropdown.dismiss()

    def proceed_to_chat(self, instance):
        if hasattr(self, 'selected_db'):
            try:
                # Set the active database URL as an environment variable
                os.environ['ACTIVE_PINECONE_INDEX_URL'] = self.selected_db['url']
                
                # Log the selection
                print(f"Proceeding to chat with database: {self.selected_db['name']}")
                print(f"Using URL: {self.selected_db['url']}")
                
                # Get the app instance to access the QASystem
                app = self.manager.get_parent_window().children[0]
                
                # Completely recreate the QASystem with the new database
                def initialize_qa(dt):
                    try:
                        # Check if app has a QA system already
                        if hasattr(app, 'qa_system') and app.qa_system is not None:
                            # Use the reinitialize method to refresh the QA system with the new database
                            print("Reinitializing existing QA system with new database")
                            app.qa_system.reinitialize()
                        else:
                            # We need to create a new QA system, but we'll let the ChatScreen handle that
                            # to avoid circular imports
                            print("QA system will be created when needed in the chat screen")
                            app.qa_system = None
                            
                        if hasattr(app, 'qa_system') and app.qa_system is not None and hasattr(app.qa_system, 'current_db_url'):
                            print(f"QA System using database URL: {app.qa_system.current_db_url}")
                        
                        # Clear chat history in the chat screen if it exists
                        if hasattr(app.sm, 'get_screen'):
                            try:
                                chat_screen = app.sm.get_screen('chat')
                                if chat_screen:
                                    chat_screen.chat_history = []
                                    if hasattr(chat_screen, 'chat_area'):
                                        chat_screen.chat_area.messages.clear_widgets()
                                    print("Chat history cleared")
                            except:
                                pass
                    except Exception as e:
                        print(f"Error reinitializing QA system: {str(e)}")
                        import traceback
                        print(traceback.format_exc())
                
                # Use Clock to avoid blocking the UI
                from kivy.clock import Clock
                Clock.schedule_once(initialize_qa, 0.1)
                
                # Transition to the chat screen
                self.manager.current = 'chat'
            except Exception as e:
                self.show_error(f"Error preparing chat: {str(e)}")
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