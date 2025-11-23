# Fix PyTorch DLL loading issue on Windows
import os
import platform
if platform.system() == "Windows":
    import ctypes
    from importlib.util import find_spec
    try:
        if (spec := find_spec("torch")) and spec.origin and os.path.exists(
            dll_path := os.path.join(os.path.dirname(spec.origin), "lib", "c10.dll")
        ):
            ctypes.CDLL(os.path.normpath(dll_path))
    except Exception:
        pass

from huggingface_hub import InferenceClient
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from statistics import mean
import os
from dotenv import load_dotenv
from transformers import pipeline

# Load environment variables
load_dotenv()

class SentimentEngine:
    def __init__(self):
        self.use_vader = False
        # Primary Multilingual Model (API)
        self.model_id = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
        
        # Specialized Hinglish Model (Local Pipeline)
        print("Initializing local Hinglish model (this may take a moment)...")
        try:
            self.hinglish_pipe = pipeline("text-classification", model="pascalrai/hinglish-twitter-roberta-base-sentiment")
            self.has_hinglish_model = True
        except Exception as e:
            print(f"Warning: Could not load local Hinglish model: {e}")
            self.has_hinglish_model = False
        
        # Hinglish indicators (common words)
        self.hinglish_words = {
            'kya', 'kyu', 'kaise', 'kese', 'tha', 'thi', 'the', 'hai', 'h', 'ho', 
            'bhai', 'matlab', 'nhi', 'nahi', 'acha', 'bura', 'sahi', 'galat', 
            'pyaar', 'pasand', 'mujhe', 'tu', 'tum', 'hum', 'main', 'mera', 'meri', 
            'karna', 'kar', 'raha', 'rahi', 'dekh', 'sun', 'bol', 'heh', 'pasandh'
        }
        
        # Try to load token
        token = os.getenv("HF_TOKEN")
        
        if token:
            print(f"Initializing Hugging Face API with token from .env...")
            self.client = InferenceClient(token=token)
        else:
            print("No HF_TOKEN found in .env. Using anonymous access (may be rate limited or restricted).")
            self.client = InferenceClient()

        # Initialize VADER as fallback
        try:
            nltk.data.find('sentiment/vader_lexicon.zip')
        except LookupError:
            nltk.download('vader_lexicon', quiet=True)
        self.vader = SentimentIntensityAnalyzer()

    def _is_hinglish(self, text):
        """
        Detects if text is Hinglish (Hindi-English code-mixing in Latin script).
        Uses langdetect + script detection + keyword matching.
        """
        try:
            from langdetect import detect, LangDetectException
            
            # Detect language
            try:
                lang = detect(text)
            except LangDetectException:
                # If detection fails, fall back to keyword matching
                lang = None
            
            # Check if text contains Devanagari script (pure Hindi)
            has_devanagari = any('\u0900' <= char <= '\u097F' for char in text)
            
            # If it's pure Devanagari Hindi, it's NOT Hinglish
            if has_devanagari:
                return False
            
            # If langdetect says it's Hindi but no Devanagari, it's likely Hinglish (Romanized)
            if lang == 'hi':
                return True
            
            # Additional check: if English but contains Hindi keywords, it's Hinglish
            if lang == 'en':
                words = set(text.lower().split())
                if words.intersection(self.hinglish_words):
                    return True
            
            return False
            
        except ImportError:
            # Fallback to simple keyword matching if langdetect not installed
            print("Warning: langdetect not installed. Using keyword-based detection.")
            words = set(text.lower().split())
            return bool(words.intersection(self.hinglish_words))

    def analyze_statement(self, text):
        """
        Analyzes a single statement. Detects Hinglish vs Standard.
        """
        if not self.use_vader:
            try:
                # Detect Language
                if self.has_hinglish_model and self._is_hinglish(text):
                    print(f"DEBUG: Detected Hinglish. Using local pipeline.")
                    
                    # Local Pipeline Inference
                    result = self.hinglish_pipe(text)[0]
                    label = result['label'].lower()
                    score = result['score']
                    
                    print(f"DEBUG: Local Hinglish Model Output -> Label: {label}, Score: {score}")

                    # Hinglish Model Mapping (positive, negative, neutral)
                    # The model outputs 'positive', 'negative', 'neutral' directly based on your snippet
                    if 'positive' in label:
                        compound = score
                        display_label = 'Positive'
                    elif 'negative' in label:
                        compound = -score
                        display_label = 'Negative'
                    else: # neutral
                        compound = 0
                        display_label = 'Neutral'
                        
                    return {
                        'text': text,
                        'scores': {'pos': score if 'positive' in label else 0, 'neg': score if 'negative' in label else 0},
                        'compound': compound,
                        'label': display_label
                    }
                
                else:
                    print(f"DEBUG: Detected Standard/Multilingual. Using API model: {self.model_id}")
                    
                    # HF Inference API
                    response = self.client.text_classification(text, model=self.model_id)
                    top_result = response[0]
                    
                    label = top_result.label.lower()
                    score = top_result.score

                    print(f"DEBUG: Hugging Face API Output -> Label: {top_result.label}, Score: {score}")

                    # Multilingual Model Mapping (Existing Logic)
                    if 'positive' in label:
                        if score > 0.9:
                            compound = score
                            display_label = 'Very Positive'
                        else:
                            compound = score * 0.5
                            display_label = 'Positive'
                    elif 'negative' in label:
                        if score > 0.9:
                            compound = -score
                            display_label = 'Very Negative'
                        else:
                            compound = -score * 0.5
                            display_label = 'Negative'
                    elif 'neutral' in label:
                        compound = 0
                        display_label = 'Neutral'
                    else:
                        compound = 0
                        display_label = label.title()

                    return {
                        'text': text,
                        'scores': {'pos': score if display_label in ['Positive', 'Very Positive'] else 0, 
                                   'neg': score if display_label in ['Negative', 'Very Negative'] else 0},
                        'compound': compound,
                        'label': display_label
                    }
            
            except Exception as e:
                print(f"Error calling HF API: {e}")
                print("Falling back to VADER for this session.")
                self.use_vader = True
        
        # Fallback: VADER
        return self._analyze_vader(text)

    def _analyze_vader(self, text):
        scores = self.vader.polarity_scores(text)
        compound = scores['compound']
        
        label = 'Neutral'
        if compound >= 0.05:
            label = 'Positive'
        elif compound <= -0.05:
            label = 'Negative'
            
        print(f"DEBUG: VADER Output -> {label} ({compound})")
        
        return {
            'text': text,
            'scores': scores,
            'compound': compound,
            'label': label
        }

    def analyze_conversation(self, statements):
        """
        Analyzes the overall sentiment of a list of statement scores.
        """
        if not statements:
            return {
                'compound': 0,
                'label': 'Neutral',
                'trend': 'No data'
            }
        
        compounds = [s['compound'] for s in statements]
        avg_compound = mean(compounds)
        
        # Determine trend
        trend = "Stable"
        if len(compounds) > 1:
            first_half = compounds[:len(compounds)//2]
            second_half = compounds[len(compounds)//2:]
            if not first_half or not second_half:
                pass # Not enough data
            else:
                mean_first = mean(first_half)
                mean_second = mean(second_half)
                if mean_second - mean_first > 0.2:
                    trend = "Improving"
                elif mean_first - mean_second > 0.2:
                    trend = "Declining"

        return {
            'compound': avg_compound,
            'label': self._get_label(avg_compound),
            'trend': trend
        }

    def _get_label(self, compound_score):
        if compound_score >= 0.1:
            return 'Positive'
        elif compound_score <= -0.1:
            return 'Negative'
        else:
            return 'Neutral'
