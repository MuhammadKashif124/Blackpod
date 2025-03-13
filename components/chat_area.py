from kivy.uix.scrollview import ScrollView
from kivy.uix.gridlayout import GridLayout
from kivy.clock import Clock
from .message_bubble import MessageBubble

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