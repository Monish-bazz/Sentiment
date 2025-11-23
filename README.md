# Sentiment Analysis Chatbot
This project delivers an intelligent, multilingual chatbot capable of natural, context-aware conversations while performing both statement-level and conversation-level sentiment analysis. Built as a modular pipeline, it supports 20+ languages—including Hindi and Hinglish with strong code-mix handling—and ensures smooth operation through a robust fallback mechanism even when Hugging Face services fail. Designed for real-world use, the system offers consistent, fault-tolerant performance, accurate emotional insights, and seamless integration into broader AI applications.


## Features

### Conversation-Level Sentiment Analysis
- Maintains full conversation history.
- Generates an overall sentiment analysis (Positive, Negative, Neutral) at the end of the interaction.

### Statement-Level Sentiment Analysis
- Evaluates sentiment for every user message individually.
- Displays the sentiment label and score alongside each message.
- **Bonus**: Summarizes the trend in mood (e.g., Improving, Declining, Stable) across the conversation.

## Technologies Used

- **Python 3**: Core programming language.
- **Hugging Face Inference API**:
  - **Primary**: `tabularisai/multilingual-sentiment-analysis` (22 languages: English, Chinese, Spanish, Hindi, Arabic, Bengali, Portuguese, Russian, Japanese, German, Malay, Telugu, Vietnamese, Korean, French, Turkish, Italian, Polish, Ukrainian, Tagalog, Dutch, Swiss German, and Swahili).
  - **Hinglish**: `pascalrai/hinglish-twitter-roberta-base-sentiment` (Specialized for Hindi-English code-mixing).
- **FastAPI**: For the web interface backend.

## Setup and Execution

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: The application will automatically download the necessary `vader_lexicon` from NLTK on the first run as a fallback.*

2. **Run the Chatbot**:
   ```bash
   python main_api.py
   ```
   Then open your browser and navigate to `http://127.0.0.1:8000`.
   
   - **Hugging Face Token (Recommended)**:
     To avoid rate limits or 401 errors with the Inference API, get a free token from [Hugging Face Settings](https://huggingface.co/settings/tokens).
     Create a `.env` file in the root directory and add:
     ```
     HF_TOKEN=your_token_here
     ```
     If no token is provided or the API fails, the system will automatically fall back to **VADER** (Tier 1 logic).

3. **Run Tests**:
   ```bash
   python -m unittest discover tests
   ```

## Explanation of Sentiment Logic

### Dual-Model Architecture

The system uses an intelligent **dual-model pipeline** that automatically selects the best model based on the input language:

#### 1. Language Detection
The system uses a **multi-layered detection approach** to accurately identify Hinglish:

1. **Statistical Language Detection**: Uses `langdetect` library to identify the base language
2. **Script Detection**: Checks for Devanagari characters (\u0900-\u097F)
3. **Keyword Matching**: Validates with common Hinglish words

**Detection Logic**:
- **Pure Hindi (Devanagari)**: Contains Devanagari script → Use multilingual model
- **Hinglish (Romanized)**: Detected as Hindi but in Latin script → Use Hinglish model
- **English with Hindi words**: Detected as English but contains Hindi keywords → Use Hinglish model
- **Pure English**: No Hindi indicators → Use multilingual model

**Examples**:
- "tu mujhe pasandh heh" → Detected as `hi` (Hindi) + Latin script → **Hinglish** ✓
- "yeh bahut acha hai bhai" → Detected as `hi` + Latin script → **Hinglish** ✓
- "I am happy" → Detected as `en` + no keywords → **English** ✓
- "मुझे यह पसंद है" → Detected as `hi` + Devanagari → **Hindi** ✓

#### 2. Model Selection

**For Hinglish Input**:
- **Model**: Local `pascalrai/hinglish-twitter-roberta-base-sentiment` (runs via transformers pipeline)
- **Output**: `positive`, `negative`, or `neutral` with confidence score
- **Mapping**: Direct mapping to sentiment labels

**For Standard Input (English, Hindi, Spanish, etc.)**:
- **Model**: Hugging Face Inference API with `tabularisai/multilingual-sentiment-analysis`
- **Output**: 5 explicit classes: `very negative`, `negative`, `neutral`, `positive`, `very positive`
- **Mapping**: 
  - `very positive` → **Very Positive** (compound: +1.0 × score)
  - `positive` → **Positive** (compound: +0.5 × score)
  - `neutral` → **Neutral** (compound: 0.0)
  - `negative` → **Negative** (compound: -0.5 × score)
  - `very negative` → **Very Negative** (compound: -1.0 × score)

#### 3. Conversation Analysis
- **Overall Sentiment**: Calculated by averaging the compound scores of all user messages
- **Trend Analysis**: Compares the average sentiment of the first half vs. the second half to detect mood shifts (Improving/Declining/Stable)

## Project Structure

- `src/sentiment.py`: Core logic for sentiment analysis using NLTK.
- `src/chatbot.py`: Manages conversation state, history, and bot responses.
- `main_api.py`: FastAPI backend application.
- `tests/`: Unit tests for the application.

## Highlights of Innovations & Enhancements

1.  **Hybrid Sentiment Engine**:
    - The system implements a robust **fallback mechanism**. It prioritizes the state-of-the-art **Hugging Face Inference API** (Deep Learning) for maximum accuracy.
    - If the API is unreachable (rate limits, network issues, or missing token), it seamlessly switches to **NLTK VADER** (Rule-based) without crashing. This ensures high availability and reliability.

2.  **Mood Trend Detection**:
    - Beyond simple averaging, the bot analyzes the **trajectory** of the conversation (First Half vs. Second Half) to determine if the user's mood is *Improving*, *Declining*, or *Stable*.

3.  **Modern Responsive UI**:
    - A clean, professional web interface built with **FastAPI** and vanilla CSS (Glassmorphism design).
    - Features real-time sentiment feedback (color-coded badges) and a modal for final detailed analysis.

4.  **Secure Configuration**:
    - Uses `.env` for secure token management, following industry best practices for handling API keys.

5.  **Interactive Sentiment Graph**:
    - A dynamic line chart (using Chart.js) visualizes the emotional trajectory of the conversation, allowing users to see exactly when the mood shifted.

6.  **Multilingual & Hinglish Support**:
    - **Dual-Model Pipeline**: The system intelligently detects the language of the input.
    - **Standard**: Uses a multilingual DistilBERT model for English, Hindi, Spanish, etc.
    - **Hinglish**: Automatically switches to a specialized `roberta-base` model when it detects Hindi-English code-mixing (e.g., "kya haal hai bhai"), ensuring accurate sentiment analysis for informal Indian text.

## Future Improvements

### Voice-Based Sentiment Analysis
The current implementation uses text-based input for sentiment analysis. A powerful enhancement would be to integrate **voice-based interaction** using **LiveKit**, enabling real-time voice conversations with sentiment analysis.

**Proposed Architecture**:
1. **Voice Input**: Use LiveKit's real-time audio streaming to capture user speech
2. **Speech-to-Text**: Integrate a multilingual STT service (e.g., Whisper, Google Speech-to-Text) to transcribe audio to text
3. **Sentiment Analysis**: Process the transcribed text through the existing dual-model pipeline
4. **Voice Response**: Use Text-to-Speech (TTS) to provide audio feedback to the user
5. **Real-time Visualization**: Display live sentiment scores and conversation trends during the voice call

**Benefits**:
- More natural and accessible interaction
- Support for users who prefer voice over typing
- Ability to analyze tone and prosody (future enhancement)
- Real-time sentiment feedback during conversations
- Multilingual voice support (English, Hindi, Hinglish)

**Technical Stack**:
- **LiveKit**: For real-time audio/video communication
- **Whisper/Google STT**: For accurate multilingual speech recognition
- **Existing Pipeline**: Reuse the current sentiment analysis logic
- **ElevenLabs/Google TTS**: For natural-sounding voice responses

This would transform the chatbot into a fully interactive voice assistant capable of understanding and analyzing emotions in real-time conversations.
