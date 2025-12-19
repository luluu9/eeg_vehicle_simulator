import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

from src.predictor.core.classifiers import CSPSVMClassifier, TGSPClassifier

def test_train():
    print("Testing TGSPClassifier.train()...")
    clf = TGSPClassifier(r"eeg_vehicle_simulator\src\predictor\models\csp_svm.joblib")
    
    try:
        clf.train()
        print("Training successful!")
    except Exception as e:
        print(f"Training failed as expected (or unexpected): {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_train()
