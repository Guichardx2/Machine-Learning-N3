from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, roc_curve, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import pandas as pd

def load_data():
    data = pd.read_csv('data/creditcard.csv')
    X = data.drop('Class', axis=1)
    y = data['Class']

    # Create a new combined feature for V16, V17, and V18
    X['V_combined'] = (X['V16'] + X['V17'] + X['V18'])

    # Drop the original V16, V17, and V18 columns
    X = X.drop(['V16', 'V17', 'V18', 'V24', 'V22', 'V25'], axis=1)

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Handle class imbalance with SMOTE
    smote = SMOTE()
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

    return X_resampled, y_resampled, X.columns

# Plot ROC curve
def plot_roc_curve(y_test, y_proba):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_score(y_test, y_proba))
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

# Plot correlation matrix
def plot_correlation_matrix(X, columns):
    plt.figure(figsize=(12, 10))
    correlation_matrix = pd.DataFrame(X, columns=columns).corr()
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', square=True, xticklabels=columns, yticklabels=columns)
    plt.title('Correlation Matrix')
    plt.show()

# Plot confusion matrix
def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Fraud', 'Fraud'], yticklabels=['Not Fraud', 'Fraud'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

# Create, train, and evaluate the model
def train_model():
    # Load and preprocess data
    X_resampled, y_resampled, columns = load_data()

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    # Initialize the Logistic Regression model
    model = LogisticRegression(class_weight='balanced', random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Evaluate the model
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_proba))

    # Plot ROC curve
    plot_roc_curve(y_test, y_proba)

    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred)

    plot_correlation_matrix(X_resampled, columns)

train_model()
