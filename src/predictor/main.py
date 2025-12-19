import sys
from PyQt6.QtWidgets import QApplication
from .ui.main_window import PredictorWindow
import os
import glob
from .core.classifiers import CSPSVMClassifier, TGSPClassifier

def main():
    app = QApplication(sys.argv)
    window = PredictorWindow()
    
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    possible_paths = glob.glob(os.path.join(models_dir, "*.pkl")) + glob.glob(os.path.join(models_dir, "*.joblib"))
    
    seen_names = set()
    for p in possible_paths:
        try:
            if p in seen_names: continue
            # Just a heuristic to guess it's a model we can load
            if "csp" in p.lower() or "svm" in p.lower():
                seen_names.add(p)
                clf = CSPSVMClassifier(p)
                window.add_classifier_ui(clf)
            if "tgsp" in p.lower():
                seen_names.add(p)
                clf = TGSPClassifier(p)
                window.add_classifier_ui(clf)
        except Exception:
            pass
            
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
