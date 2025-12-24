import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
data_dict = pickle.load(open('data.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])
print("Feature count:", data.shape[1])  # MUST be 42
x_train, x_test, y_train, y_test = train_test_split(
    data,
    labels,
    test_size=0.2,
    stratify=labels,
    random_state=42
)
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
print("✅ model.p saved")