import itertools
import sys
import threading
import time
from enum import Enum
from typing import ContextManager, Union
from ui.base import BaseHumanUserInterface


class Color(Enum):
    """Color codes for the commandline"""
    BLACK = '\033[30m'  # (Text) Black
    RED = '\033[31m'  # (Text) Red
    GREEN = '\033[32m'  # (Text) Green
    YELLOW = '\033[33m'  # (Text) Yellow
    BLUE = '\033[34m'  # (Text) Blue
    MAGENTA = '\033[35m'  # (Text) Magenta
    CYAN = '\033[36m'  # (Text) Cyan
    WHITE = '\033[37m'  # (Text) White
    COLOR_DEFAULT = '\033[39m'  # Reset text color to default


class CommandlineUserInterface(BaseHumanUserInterface):
    """Commandline user interface."""

    def get_user_input(self) -> str:
        """Get user input and return the result as a string"""
        user_input = input("Input:")
        return str(user_input)

    def get_binary_user_input(self, prompt: str) -> bool:
        """Get a binary input from the user and return the result as a bool"""
        yes_patterns = ["y", "yes", "yeah", "yup", "yep"]
        no_patterns = ["n", "no", "nah", "nope"]
        while True:
            response = input(prompt + " (y/n) ").strip().lower()
            if response in yes_patterns:
                return True
            elif response in no_patterns:
                return False
            else:
                self.notify("Invalid input", "Please enter y or n.",
                            title_color=Color.RED)
                continue

    def notify(self, title: str, message: str, title_color: Union[str, Color] = Color.YELLOW) -> None:
        """Print a notification to the user"""
        if isinstance(title_color, str):
            try:
                title_color = Color[title_color.upper()]
            except KeyError:
                raise ValueError(f"{title_color} is not a valid Color")
        self._print_message(title, message, title_color)

    def loading(self,
                message: str = "Thinking...",
                delay: float = 0.1) -> ContextManager:
        """Return a context manager that will display a loading spinner"""
        return self.Spinner(message=message, delay=delay)

    def _print_message(self, title: str, message: str, title_color: Color) -> None:
        print(f"{title_color.value}{title}{Color.COLOR_DEFAULT.value}: {message}")

    class Spinner:
        """A simple spinner class"""

        def __init__(self, message="Loading...", delay=0.1):
            """Initialize the spinner class"""
            self.spinner = itertools.cycle(['-', '/', '|', '\\'])
            self.delay = delay
            self.message = message
            self.running = False
            self.spinner_thread = None

        def spin(self):
            """Spin the spinner"""
            while self.running:
                sys.stdout.write(next(self.spinner) + " " + self.message + "\r")
                sys.stdout.flush()
                time.sleep(self.delay)
                sys.stdout.write('\b' * (len(self.message) + 2))

        def __enter__(self):
            """Start the spinner"""
            self.running = True
            self.spinner_thread = threading.Thread(target=self.spin)
            self.spinner_thread.start()

        def __exit__(self, exc_type, exc_value, exc_traceback):
            """Stop the spinner"""
            self.running = False
            self.spinner_thread.join()
            sys.stdout.write('\r' + ' ' * (len(self.message) + 2) + '\r')
            sys.stdout.flush()
