from kivy.uix.button import Button
from kivy.graphics import Color, RoundedRectangle, Line

class RoundedButton(Button):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.background_color = [0, 0, 0, 0]
        self.background_normal = ''
        self.font_name = 'Roboto'
        self.color = [1, 1, 1, 1]  # White text
        
        with self.canvas.before:
            self.bg_color = Color(rgba=(0.25, 0.5, 1, 1))  # Store Color instruction
            self.rect = RoundedRectangle(pos=self.pos, size=self.size, radius=[25])
        
        with self.canvas.after:
            Color(rgba=(1, 1, 1, 0.1))
            self._border = Line(rounded_rectangle=(self.pos[0], self.pos[1], self.size[0], self.size[1], 25), width=1.2)
        
        self.bind(pos=self._update_rect, size=self._update_rect)
        self.bind(state=self._update_color)
    
    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size
        self._border.rounded_rectangle = (instance.pos[0], instance.pos[1], instance.size[0], instance.size[1], 25)
    
    def _update_color(self, instance, value):
        if value == 'down':
            self.bg_color.rgba = (0.2, 0.4, 0.9, 1)  # Darker blue when pressed
        else:
            self.bg_color.rgba = (0.25, 0.5, 1, 1)  # Normal blue