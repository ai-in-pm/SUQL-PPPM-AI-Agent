from abc import ABC, abstractmethod

class BaseAgent(ABC):
    """
    Abstract base class for all AI agents in the SUQL-PPM system.
    """
    def __init__(self, name, description):
        self.name = name
        self.description = description
        self.is_initialized = False
    
    def initialize(self):
        """
        Initialize the agent with necessary resources.
        """
        self._initialize_resources()
        self.is_initialized = True
        print(f"{self.name} initialized successfully.")
    
    @abstractmethod
    def _initialize_resources(self):
        """
        Initialize agent-specific resources. To be implemented by subclasses.
        """
        pass
    
    @abstractmethod
    def process(self, input_data):
        """
        Process the input data and return the output. To be implemented by subclasses.
        """
        pass
    
    def __str__(self):
        return f"{self.name}: {self.description}"
