import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, LabelBinarizer

# Load datasets
df = pd.read_csv("Training.csv")
tr = pd.read_csv("Testing.csv")

# Encode labels to integers
label_encoder = LabelEncoder()
df['prognosis'] = label_encoder.fit_transform(df['prognosis'])
tr['prognosis'] = label_encoder.transform(tr['prognosis'])

# Extract features and labels
X = df.iloc[:, :-1]
y = df['prognosis']
X_test = tr.iloc[:, :-1]
y_test = tr['prognosis']

# Train SVM model
clf = SVC(probability=True, kernel='linear', random_state=42)
clf.fit(X, y)

# Predictions
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Classification Report
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
print("Classification Report:\n", report)

# Save Classification Report
with open("classification_report.txt", "w") as file:
    file.write("Classification Report:\n")
    file.write(report)

    
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Save Confusion Matrix plot
plt.figure(figsize=(12, 10))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.xticks(np.arange(len(label_encoder.classes_)), label_encoder.classes_, rotation=90)
plt.yticks(np.arange(len(label_encoder.classes_)), label_encoder.classes_)
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()

# ROC Curve
lb = LabelBinarizer()
y_test_binarized = lb.fit_transform(y_test)
fpr, tpr, roc_auc = {}, {}, {}

for i in range(len(label_encoder.classes_)):
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Save ROC Curve plot
plt.figure(figsize=(12, 10))
for i in range(len(label_encoder.classes_)):
    plt.plot(fpr[i], tpr[i], label=f'{label_encoder.classes_[i]} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', label='Chance')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right', bbox_to_anchor=(1.5, 0.5), fontsize='small')
plt.tight_layout()
plt.savefig("roc_curve.png")
plt.close()

