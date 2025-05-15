import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import os

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

class SpamHamClassifier:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.tfidf_vectorizer = None
        self.model = None
        self.feature_log_probs = None
        self.feature_names = None
        self.classes = None
        self.class_priors = {'spam': 0.15, 'ham': 0.85}  
    
    def preprocess_text(self, text):
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)  
        
        tokens = word_tokenize(text)
        
        tokens = [word.lower() for word in tokens]
        tokens = [word for word in tokens if word not in self.stop_words and len(word) > 1]
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        
        return tokens
    
    def train(self, file_path):
        df = pd.read_csv(file_path, sep=";", encoding="windows-1252", names=["label", "message"], header=0)
        df['label'] = df['label'].str.strip()
        df = df[df['label'].isin(['ham', 'spam'])]
        df.dropna(subset=["message"], inplace=True)
        
        df['tokens'] = df['message'].apply(self.preprocess_text)
        df['clean_text'] = df['tokens'].apply(lambda x: ' '.join(x))
        
        label_counts = df['label'].value_counts(normalize=True)
        self.class_priors = {'spam': label_counts.get('spam', 0.15), 
                            'ham': label_counts.get('ham', 0.85)}
        
        print(f"Class priors calculados del dataset: {self.class_priors}")
        
        X_train, X_test, y_train, y_test = train_test_split(
            df['clean_text'], 
            df['label'], 
            test_size=0.2, 
            random_state=42,
            stratify=df['label']
        )
        
        self.tfidf_vectorizer = TfidfVectorizer(
            min_df=5,
            max_df=0.8,
            sublinear_tf=True,
            use_idf=True,
            norm='l2'
        )
        
        X_train_tfidf = self.tfidf_vectorizer.fit_transform(X_train)
        X_test_tfidf = self.tfidf_vectorizer.transform(X_test)
        
        self.model = MultinomialNB()
        self.model.fit(X_train_tfidf, y_train)
        
        self.feature_names = self.tfidf_vectorizer.get_feature_names_out()
        self.feature_log_probs = self.model.feature_log_prob_
        self.classes = self.model.classes_
        
        y_pred = self.model.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)
        
        print("\n--- RESULTADOS DE EVALUACIÓN ---")
        print(f"Exactitud: {accuracy:.4f}")
        print("\nMatriz de Confusión:")
        print(conf_matrix)
        print("\nReporte de Clasificación:")
        print(class_report)
        
        return accuracy, conf_matrix, class_report
    
    def save_model(self, model_path="spam_ham_classifier.pkl"):
        with open(model_path, 'wb') as f:
            pickle.dump({
                'vectorizer': self.tfidf_vectorizer,
                'model': self.model,
                'feature_names': self.feature_names,
                'classes': self.classes,
                'class_priors': self.class_priors
            }, f)
        print(f"Modelo guardado en {model_path}")
    
    def load_model(self, model_path="spam_ham_classifier.pkl"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"El archivo {model_path} no existe")
        
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
            self.tfidf_vectorizer = data['vectorizer']
            self.model = data['model']
            self.feature_names = data['feature_names']
            self.classes = data['classes']
            self.feature_log_probs = self.model.feature_log_prob_
            self.class_priors = data.get('class_priors', {'spam': 0.15, 'ham': 0.85})
        print(f"Modelo cargado desde {model_path}")
    
    def clasificar_texto(self, text):
        if not self.model or not self.tfidf_vectorizer:
            raise ValueError("El modelo no ha sido entrenado o cargado")
        
        tokens = self.preprocess_text(text)
        clean_text = ' '.join(tokens)
        
        text_tfidf = self.tfidf_vectorizer.transform([clean_text])
        
        probabilities = self.model.predict_proba(text_tfidf)[0]
        spam_index = np.where(self.model.classes_ == 'spam')[0][0]
        spam_probability = probabilities[spam_index]
        
        spam_class_index = list(self.classes).index('spam')
        ham_class_index = list(self.classes).index('ham')
        
        p_spam = self.class_priors['spam']
        p_ham = self.class_priors['ham']
        
        word_indices = {}
        for word in tokens:
            try:
                word_idx = np.where(self.feature_names == word)[0]
                if len(word_idx) > 0:
                    word_indices[word] = word_idx[0]
            except:
                continue
        
        word_predictive_power = {}
        for word, idx in word_indices.items():
            p_word_spam = np.exp(self.feature_log_probs[spam_class_index, idx])
            p_word_ham = np.exp(self.feature_log_probs[ham_class_index, idx])
            
            p_spam_given_word = (p_word_spam * p_spam) / ((p_word_spam * p_spam) + (p_word_ham * p_ham))
            word_predictive_power[word] = p_spam_given_word
        
        if not word_predictive_power:
            top_predictive_words = []
        else:
            top_predictive_words = sorted(word_predictive_power.items(), key=lambda x: x[1], reverse=True)[:3]
        
        prediction = 'spam' if spam_probability > 0.5 else 'ham'
        
        return {
            'prediction': prediction,
            'spam_probability': spam_probability,
            'top_spam_words': top_predictive_words
        }

def clasificar_prompt(model_path="spam_ham_classifier.pkl"):
    classifier = SpamHamClassifier()    
    try:
        classifier.load_model(model_path)
    except FileNotFoundError:
        print("Modelo no encontrado. Entrenando nuevo modelo...")
        print("Asegúrate de tener el archivo spam_ham.csv en el directorio actual")
        classifier.train("spam_ham.csv")
        classifier.save_model(model_path)
    
    while True:
        text = input("\nIngresa el texto a clasificar (o 'q' para salir): ")
        
        if text.lower() == 'q':
            break
        
        result = classifier.clasificar_texto(text)
        
        print("\n--- RESULTADOS DE CLASIFICACIÓN ---")
        print(f"Clasificación: {result['prediction'].upper()}")
        print(f"Probabilidad de SPAM: {result['spam_probability']*100:.2f}%")
        
        if result['top_spam_words']:
            print("\nPalabras con mayor poder predictivo de SPAM:")
            for word, prob in result['top_spam_words']:
                print(f"- {word}: {prob*100:.2f}%")
        else:
            print("\nNo se encontraron palabras con poder predictivo en el vocabulario del modelo")
    
    return result

if __name__ == "__main__":
    clasificar_prompt()