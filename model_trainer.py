from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from tools import get_training_data
from mne.decoding import Vectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from tools import get_freq



train_epochs, y_train, test_epochs, y_test, valid_epochs, y_valid = get_training_data()
X_train, freqs = get_freq(train_epochs)
X_valid, _ = get_freq(valid_epochs)
X_test, _ = get_freq(test_epochs)

clf = Pipeline([
    ('vectorizer', Vectorizer()),
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        n_jobs=-1,
        class_weight='balanced'
    ))
])

clf.fit(X_train, y_train)

y_pred_valid = clf.predict(X_valid)
print("VALID:")
print(classification_report(y_valid, y_pred_valid))
print(confusion_matrix(y_valid, y_pred_valid))

y_pred_test = clf.predict(X_test)
print("TEST:")
print(classification_report(y_test, y_pred_test))
print(confusion_matrix(y_test, y_pred_test))

joblib.dump(clf, 'model.joblib')
