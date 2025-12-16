from abc import ABC, abstractmethod
import joblib
import numpy as np
import scipy.signal as signal
from ...common.constants import LSLChannel

class BaseClassifier(ABC):
    def __init__(self):
        self._min_window = 1.0 # Default
        self._max_window = 5.0 # Default
    
    @property
    def min_window(self) -> float:
        """Minimum required window duration in seconds."""
        return self._min_window
        
    @property
    def max_window(self) -> float:
        """Maximum supported window duration in seconds."""
        return self._max_window
        
    @abstractmethod
    def predict_proba(self, data: np.ndarray, fs: float) -> np.ndarray:
        """
        Predict probabilities for the class set:
        [Relax, Left, Right, Both, Feet]
        
        Args:
            data: EEG data (n_channels, n_samples)
            fs: Sampling rate of the data
            
        Returns:
            np.ndarray: Probabilities array of shape (5,)
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Display name of the classifier."""
        pass

class MockClassifier(BaseClassifier):
    def __init__(self):
        super().__init__()
        self._min_window = 0.5
        self._max_window = 10.0
        
    @property
    def name(self):
        return "MockGeneric"
        
    def predict_proba(self, data: np.ndarray, fs: float) -> np.ndarray:
        # Return random probabilities normalized to sum 1
        probs = np.random.dirichlet(np.ones(5), size=1)[0]
        return probs

class CSPSVMClassifier(BaseClassifier):
    def __init__(self, model_path: str):
        super().__init__()
        self.model_path = model_path
        self._min_window = 2.0
        self._max_window = 5.0
        
        try:
            self.model = joblib.load(self.model_path)
            print(f"Loaded model: {model_path}")
        except Exception as e:
            print(f"Failed to load model {model_path}: {e}")
            raise

    @property
    def name(self):
        return "CSP+SVM"

    def predict_proba(self, data: np.ndarray, fs: float) -> np.ndarray:
        # Prepare for prediction (1, ch, time)
        X = data[np.newaxis, :, :] 
        
        try:
            probs = self.model.predict_proba(X)[0] 
            classes = self.model.classes_ # e.g. [1, 2] or [2, 3, 4, 5], where 1=Relax, 2=Left, etc.
            
            # Map to standard vector of size 5 even if the model has different number of classes
            # classes-1 because the classes are 1-based, and we want 0-based indexing
            full_probs = np.zeros(5)
            full_probs[classes-1] = probs
            
            return full_probs
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return np.zeros(5)
