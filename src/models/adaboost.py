import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,accuracy_score, roc_auc_score
from sklearn.discriminant_analysis import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)

def load_data():
    data = pd.read_csv('data/raw/creditcard.csv')
    X = data.drop('Class', axis=1)
    y = data['Class']
    return X, y

def plot_correlation_matrix(X):
    plt.figure(figsize=(12, 10))
    correlation_matrix = pd.DataFrame(X).corr()
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', square=True, xticklabels=True, yticklabels=True)
    plt.title('Correlation Matrix')
    plt.show()

def train_model():
    print("Starting...")
    X, y = load_data()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    smote = SMOTE()
    X_train, y_train = smote.fit_resample(X_train, y_train)

    base_estimator = DecisionTreeClassifier(max_depth=1)

    # Hyperparameter tuning with GridSearchCV
    param_grid = {
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.01, 0.1, 1.0]
    }

    grid_search = GridSearchCV(estimator=AdaBoostClassifier(estimator=base_estimator, random_state=42, algorithm="SAMME"),
                               param_grid=param_grid,
                               scoring='f1',
                               cv=3,
                               n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Best model
    best_model = grid_search.best_estimator_
    print("Melhores parâmetros:", grid_search.best_params_)

    # Predictions
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    # Metrics
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_proba))

    # Plots
    # plot_roc_curve(y_test, y_proba)
    # plot_confusion_matrix(y_test, y_pred)
    plot_correlation_matrix(X)

    print("Finalizado.")

train_model()
