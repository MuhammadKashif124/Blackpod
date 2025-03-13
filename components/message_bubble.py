from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.graphics import Color, RoundedRectangle
from kivy.animation import Animation

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