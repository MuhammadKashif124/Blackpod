import os
import sys
import logging
from kivy.uix.screenmanager import Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.graphics import Color, RoundedRectangle
from kivy.app import App
import pymongo
import hashlib
from .rounded_button import RoundedButton
from kivy.uix.popup import Popup
# Import UserManager for authentication
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from user import UserManager

class LoginScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Main layout
        self.layout = BoxLayout(
            orientation='vertical',
            padding=[30, 50],
            spacing=20
        )
        
        # Add title
        self.title = Label(
            text='BlackPod AI',
            font_size='32sp',
            size_hint=(1, 0.2),
            halign='center'
        )
        self.layout.add_widget(self.title)
        
        # Add subtitle
        self.subtitle = Label(
            text='Login to your account',
            font_size='18sp',
            size_hint=(1, 0.1),
            halign='center'
        )
        self.layout.add_widget(self.subtitle)
        
        # Add spacer
        self.layout.add_widget(BoxLayout(size_hint=(1, 0.1)))
        
        # Username input
        self.username_input = TextInput(
            hint_text='Username',
            multiline=False,
            size_hint=(1, None),
            height=50,
            padding=[10, 15],
            background_color=(0.12, 0.12, 0.17, 1),
            foreground_color=(1, 1, 1, 1),
            cursor_color=(1, 1, 1, 1),
            font_size='16sp'
        )
        self.layout.add_widget(self.username_input)
        
        # Password input
        self.password_input = TextInput(
            hint_text='Password',
            multiline=False,
            password=True,
            size_hint=(1, None),
            height=50,
            padding=[10, 15],
            background_color=(0.12, 0.12, 0.17, 1),
            foreground_color=(1, 1, 1, 1),
            cursor_color=(1, 1, 1, 1),
            font_size='16sp'
        )
        self.layout.add_widget(self.password_input)
        
        # Organization Key input
        self.org_key_input = TextInput(
            hint_text='Organization Key',
            multiline=False,
            size_hint=(1, None),
            height=50,
            padding=[10, 15],
            background_color=(0.12, 0.12, 0.17, 1),
            foreground_color=(1, 1, 1, 1),
            cursor_color=(1, 1, 1, 1),
            font_size='16sp'
        )
        self.layout.add_widget(self.org_key_input)
        
        # Add spacer
        self.layout.add_widget(BoxLayout(size_hint=(1, 0.1)))
        
        # Login button
        self.login_button = RoundedButton(
            text='Login',
            size_hint=(1, None),
            height=50,
            background_color=(0.25, 0.5, 1, 1)
        )
        
        # Fix: Use on_release instead of on_press and properly handle the login action
        self.login_button.bind(on_release=self.login)
        self.layout.add_widget(self.login_button)
        
        # Add spacer
        self.layout.add_widget(BoxLayout(size_hint=(1, 0.3)))
        
        self.add_widget(self.layout)
    
    def login(self, instance):
        try:
            self.logger.info("Login button pressed")
            username = self.username_input.text.strip()
            password = self.password_input.text.strip()
            org_key = self.org_key_input.text.strip()
            
            # Log the login attempt (without the password)
            self.logger.info(f"Login attempt with username: {username}")
            
            # Validate inputs
            if not username or not password or not org_key:
                self.logger.warning("Username, password, or organization key is empty")
                self.show_error("All fields are required!")
                return
            
            # Use UserManager to authenticate
            try:
                user_manager = UserManager()
                result = user_manager.login(username, password, org_key)
                
                # Check if login was successful
                if result.startswith("Login successful"):
                    self.logger.info("Login successful, switching to database_select screen")
                    app = App.get_running_app()
                    app.sm.current = 'database_select'
                else:
                    # Show error message
                    self.logger.warning(f"Login failed: {result}")
                    self.show_error(result)
            except Exception as e:
                self.logger.error(f"Error authenticating with MongoDB: {str(e)}", exc_info=True)
                self.show_error(f"Authentication error: {str(e)}")
            
        except Exception as e:
            self.logger.error(f"Error during login: {str(e)}", exc_info=True)
            self.show_error(f"Login error: {str(e)}")

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