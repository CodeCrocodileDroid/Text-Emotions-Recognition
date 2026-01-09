import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Models
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

# Text preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-ng')


class EmotionClassifier:
    def __init__(self):
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.vectorizer = None
        self.label_encoder = None
        self.models = {}
        self.results = {}

    def load_data(self, filepath):
        """Load and preprocess the dataset"""
        self.data = pd.read_csv(filepath)

        # Display dataset info
        print("Dataset Info:")
        print(f"Shape: {self.data.shape}")
        print(f"Columns: {self.data.columns.tolist()}")
        print("\nEmotion Distribution:")
        print(self.data['Emotion'].value_counts())
        print("\nSample Data:")
        print(self.data.head())

        return self.data

    def preprocess_text(self, text):
        """Clean and preprocess text"""
        if not isinstance(text, str):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        # Remove user @ references and '#' from hashtags
        text = re.sub(r'\@\w+|\#', '', text)

        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))

        # Remove numbers
        text = re.sub(r'\d+', '', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def advanced_preprocess(self, text):
        """More advanced text preprocessing with lemmatization"""
        if not isinstance(text, str):
            return ""

        # Basic preprocessing
        text = self.preprocess_text(text)

        # Tokenization
        tokens = word_tokenize(text)

        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]

        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]

        return ' '.join(tokens)

    def prepare_data(self, use_advanced_preprocessing=True):
        """Prepare features and labels"""
        # Create a copy to avoid modifying original data
        data_copy = self.data.copy()

        # Apply preprocessing
        if use_advanced_preprocessing:
            data_copy['processed_text'] = data_copy['Text'].apply(self.advanced_preprocess)
        else:
            data_copy['processed_text'] = data_copy['Text'].apply(self.preprocess_text)

        # Remove empty texts
        data_copy = data_copy[data_copy['processed_text'].str.len() > 0]

        # Prepare features and labels
        self.X = data_copy['processed_text']
        self.y = data_copy['Emotion']

        # Encode labels
        self.label_encoder = LabelEncoder()
        self.y_encoded = self.label_encoder.fit_transform(self.y)

        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y_encoded, test_size=0.2, random_state=42, stratify=self.y_encoded
        )

        print(f"\nTraining samples: {len(self.X_train)}")
        print(f"Testing samples: {len(self.X_test)}")
        print(f"Number of classes: {len(np.unique(self.y_encoded))}")

        return self.X_train, self.X_test, self.y_train, self.y_test

    def vectorize_text(self, vectorizer_type='tfidf', max_features=5000):
        """Convert text to numerical features"""
        if vectorizer_type == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=(1, 2),  # Use unigrams and bigrams
                min_df=2,
                max_df=0.95
            )
        elif vectorizer_type == 'count':
            self.vectorizer = CountVectorizer(
                max_features=max_features,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95
            )
        else:
            raise ValueError("vectorizer_type must be 'tfidf' or 'count'")

        # Fit and transform
        X_train_vec = self.vectorizer.fit_transform(self.X_train)
        X_test_vec = self.vectorizer.transform(self.X_test)

        print(f"\nVectorized features shape - Train: {X_train_vec.shape}, Test: {X_test_vec.shape}")

        return X_train_vec, X_test_vec

    def initialize_models(self):
        """Initialize various classification models"""
        self.models = {
            'Naive Bayes': MultinomialNB(alpha=0.1),
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                C=1.0,
                random_state=42
            ),
            'SVM': SVC(
                C=1.0,
                kernel='linear',
                random_state=42,
                probability=True
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                random_state=42,
                n_jobs=-1
            ),
            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
            'XGBoost': XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                use_label_encoder=False,
                eval_metric='mlogloss'
            ),
            'Neural Network': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=1000,
                random_state=42,
                alpha=0.01
            )
        }

        return self.models

    def train_models(self, X_train_vec, X_test_vec):
        """Train all models and evaluate performance"""
        self.results = {}

        for name, model in self.models.items():
            print(f"\nTraining {name}...")

            # Train model
            model.fit(X_train_vec, self.y_train)

            # Make predictions
            y_pred = model.predict(X_test_vec)
            y_pred_proba = model.predict_proba(X_test_vec) if hasattr(model, 'predict_proba') else None

            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            report = classification_report(self.y_test, y_pred, output_dict=True)

            # Store results
            self.results[name] = {
                'model': model,
                'accuracy': accuracy,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'classification_report': report
            }

            print(f"{name} Accuracy: {accuracy:.4f}")

        return self.results

    def compare_model_performance(self):
        """Compare performance of all models"""
        print("\n" + "=" * 60)
        print("MODEL PERFORMANCE COMPARISON")
        print("=" * 60)

        performance_df = pd.DataFrame({
            'Model': list(self.results.keys()),
            'Accuracy': [self.results[model]['accuracy'] for model in self.results]
        }).sort_values('Accuracy', ascending=False)

        print(performance_df.to_string(index=False))

        # Plot comparison
        plt.figure(figsize=(12, 6))
        bars = plt.barh(performance_df['Model'], performance_df['Accuracy'], color='skyblue')
        plt.xlabel('Accuracy')
        plt.title('Model Performance Comparison')
        plt.xlim([0, 1])

        # Add accuracy values on bars
        for bar, acc in zip(bars, performance_df['Accuracy']):
            plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                     f'{acc:.4f}', va='center')

        plt.tight_layout()
        plt.show()

        return performance_df

    def get_best_model(self):
        """Get the best performing model"""
        best_model_name = max(self.results.items(), key=lambda x: x[1]['accuracy'])[0]
        best_model_info = self.results[best_model_name]

        print(f"\nBest Model: {best_model_name}")
        print(f"Best Accuracy: {best_model_info['accuracy']:.4f}")

        return best_model_name, best_model_info

    def plot_confusion_matrix(self, model_name):
        """Plot confusion matrix for a specific model"""
        if model_name not in self.results:
            print(f"Model '{model_name}' not found in results.")
            return

        y_pred = self.results[model_name]['predictions']

        # Decode labels for plotting
        y_test_decoded = self.label_encoder.inverse_transform(self.y_test)
        y_pred_decoded = self.label_encoder.inverse_transform(y_pred)

        # Get class names
        class_names = self.label_encoder.classes_

        # Create confusion matrix
        cm = confusion_matrix(y_test_decoded, y_pred_decoded, labels=class_names)

        # Plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def predict_emotion(self, text, model_name=None):
        """Predict emotion for new text"""
        if model_name is None:
            # Use best model
            model_name, _ = self.get_best_model()

        if model_name not in self.results:
            raise ValueError(f"Model '{model_name}' not found.")

        # Preprocess text
        processed_text = self.advanced_preprocess(text)

        # Vectorize
        text_vec = self.vectorizer.transform([processed_text])

        # Predict
        model = self.results[model_name]['model']
        prediction_encoded = model.predict(text_vec)[0]
        prediction_proba = model.predict_proba(text_vec)[0]

        # Decode prediction
        prediction = self.label_encoder.inverse_transform([prediction_encoded])[0]

        # Get probability for each class
        probabilities = {}
        for i, prob in enumerate(prediction_proba):
            class_name = self.label_encoder.inverse_transform([i])[0]
            probabilities[class_name] = prob

        print(f"\nText: {text}")
        print(f"Predicted Emotion: {prediction}")
        print("\nProbabilities:")
        for emotion, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
            print(f"  {emotion}: {prob:.4f}")

        return prediction, probabilities

    def analyze_important_features(self, model_name='Logistic Regression', top_n=20):
        """Analyze most important features for prediction"""
        if model_name not in self.models:
            print(f"Model '{model_name}' not available.")
            return

        model = self.models[model_name]

        # Check if model has feature importance or coefficients
        if hasattr(model, 'coef_'):
            # For linear models
            feature_names = self.vectorizer.get_feature_names_out()

            # Get coefficients for each class
            for i, class_name in enumerate(self.label_encoder.classes_):
                coefficients = model.coef_[i]

                # Get top positive and negative features
                top_positive_idx = np.argsort(coefficients)[-top_n:][::-1]
                top_negative_idx = np.argsort(coefficients)[:top_n]

                print(f"\nTop features for '{class_name}':")
                print("-" * 40)

                print("Positive indicators:")
                for idx in top_positive_idx:
                    print(f"  {feature_names[idx]}: {coefficients[idx]:.4f}")

                print("\nNegative indicators:")
                for idx in top_negative_idx:
                    print(f"  {feature_names[idx]}: {coefficients[idx]:.4f}")
                print()

        elif hasattr(model, 'feature_importances_'):
            # For tree-based models
            importances = model.feature_importances_
            feature_names = self.vectorizer.get_feature_names_out()

            # Get top important features
            top_indices = np.argsort(importances)[-top_n:][::-1]

            print(f"\nTop {top_n} important features:")
            print("-" * 40)
            for idx in top_indices:
                print(f"  {feature_names[idx]}: {importances[idx]:.4f}")
        else:
            print(f"Feature importance analysis not available for {model_name}")


def main():
    """Main function to run the emotion classification pipeline"""

    # Initialize classifier
    classifier = EmotionClassifier()

    # Load data
    print("Loading data...")
    data = classifier.load_data('emotion_dataset_raw.csv')

    # Prepare data
    print("\nPreparing data...")
    X_train, X_test, y_train, y_test = classifier.prepare_data(use_advanced_preprocessing=True)

    # Vectorize text
    print("\nVectorizing text...")
    X_train_vec, X_test_vec = classifier.vectorize_text(vectorizer_type='tfidf', max_features=3000)

    # Initialize models
    print("\nInitializing models...")
    models = classifier.initialize_models()

    # Train and evaluate models
    print("\nTraining models...")
    results = classifier.train_models(X_train_vec, X_test_vec)

    # Compare model performance
    performance_df = classifier.compare_model_performance()

    # Get best model
    best_model_name, best_model_info = classifier.get_best_model()

    # Plot confusion matrix for best model
    classifier.plot_confusion_matrix(best_model_name)

    # Analyze important features
    classifier.analyze_important_features(model_name='Logistic Regression', top_n=15)

    # Test with sample texts
    print("\n" + "=" * 60)
    print("TESTING WITH SAMPLE TEXTS")
    print("=" * 60)

    sample_texts = [
        "I am so happy today! This is the best day of my life!",
        "I feel really sad and lonely, everything seems so difficult.",
        "I'm scared about what might happen tomorrow.",
        "This makes me so angry! I can't believe it!",
        "Wow! I wasn't expecting that at all! What a surprise!",
        "This is disgusting, I can't stand looking at it."
    ]

    for text in sample_texts:
        classifier.predict_emotion(text, model_name=best_model_name)
        print("-" * 40)

    return classifier, results, performance_df


if __name__ == "__main__":
    classifier, results, performance_df = main()