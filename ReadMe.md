# BMW Chatbot

## Overview
BMW Chatbot is an intelligent car information assistant powered by **Natural Language Processing (NLP)**. It helps users find details about BMW models, check prices, get recommendations based on budget, and analyze car reviews. The project now features a modern web interface built with **Streamlit**.

Streamlit Deployed Link: https://bmw-nlp-chatbot-hamim.streamlit.app/ 


## Features
- **Answers questions about BMW cars from bmw.csv using TF-IDF + cosine similarity.**
- **Handles greetings, thanks, help, and exit intents.**
- **Uses simple NLP to detect year and fuel type and filters the dataset accordingly.**
- **Can answer aggregate questions:**
      - "average price of 3 series 2017"
      - "cheapest 5 series diesel"
      - "most expensive 1 series"
- **Logs all interactions to chat_log.txt**
- **Has a special mode to analyze sentiment + readability of a text:**
      - "analyze: <your review text>"
- **Input Validation**: Detects invalid/gibberish input and prompts for clarification.
- **Web Interface**: Clean, interactive chat UI using Streamlit (sidebar, chat, footer).

## Requirements
- Python 3.8+
- Key libraries: `streamlit`, `pandas`, `scikit-learn`, `spacy`, `textblob`, `textatistic`

## Installation

1. **Clone the repository** (if applicable) or download the files.
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: If you don't have a `requirements.txt`, install manually:*
   ```bash
   pip install streamlit pandas scikit-learn spacy textblob textatistic
   ```
3. **Download the spaCy model**:
   ```bash
   python -m spacy download en_core_web_sm
   ```

## How to Run

Launch the web application using Streamlit:

```bash
streamlit run app.py
```

This will open the chatbot in your default web browser (usually at `http://localhost:8501`).

## Example Queries

Once the chatbot is running, try asking:

### 🚗 Car Details
* "Tell me about the 2017 BMW 3 Series diesel."
* "What is the price of a 2020 BMW X5?"
* "Specs for 2019 M4."

### 💰 Budget & Recommendations
* "My budget is 15000."
* "Recommend a car under 20k."
* "Which car is best?"
* "Suggest a cheap diesel car."

### 📊 Aggregates
* "Average price of BMW 3 Series 2020."
* "Cheapest 2019 BMW X5."
* "Most expensive BMW 7 Series."

### 📝 Text Analysis
* "Analyze: This BMW 3 Series is fast and smooth but a bit expensive."


## Project Structure

```
BMW-NLP-Chatbot/
│
├── app.py                      # Streamlit web application entry point
├── BMW_Chatbot.py              # Core chatbot logic and NLP processing
├── requirements.txt            # Python dependencies
├── ReadMe.md                   # Project documentation
│
├── data/
│   └── bmw.csv                 # Dataset containing car information
│
├── logs/
│   └── chat_log.txt            # Log of all user-bot interactions
│
├── examples/
│   └── QNA_Example.md          # List of example questions for testing
│
├── tests/
│   ├── test_BMW_Chatbot.py     # Pytest test suite (60+ test cases)
│   └── pytest_results.txt      # Saved pytest results
└── ...
```

## Running Tests

To run the test suite (60+ test cases, covering all major features):

```bash
pytest tests/test_BMW_Chatbot.py > tests/pytest_results.txt
```
All tests should pass. Results are saved in `tests/pytest_results.txt`.



## How It Works

1.  **NLP Processing**: The bot uses `spaCy` to extract entities like **Year**, **Series**, and **Fuel Type** from your query.
2.  **Filtering**: It filters the dataset (`bmw.csv`) to narrow down the search space.
3.  **Similarity Matching**: It uses **TF-IDF** and **Cosine Similarity** to find the most relevant car entry matching your description.
4.  **Intent Classification**: Rule-based logic determines if you are asking for a specific car, a recommendation, or an aggregate stat (avg/min/max).
5.  **Logging**: Every user-bot interaction is saved to `chat_log.txt` for transparency and debugging.
6.  **Input Validation**: The bot checks if your input is a valid question about BMWs and prompts if not.
7.  **Error Handling**: If the dataset or a required library is missing, the bot shows a clear error message.


## Customizing & Extending

- **Dataset**: The `bmw.csv` file drives the chatbot's knowledge. Add new rows to update the bot's knowledge base without changing the code. Required columns: `model`, `year`, `price`, `transmission`, `mileage`, `fuelType`, `tax`, `mpg`, `engineSize`.
- **Extensibility**: The code is modular. You can add new intents (e.g., "compare two cars") or swap in a new CSV for another car brand with minimal changes.


## Troubleshooting & Error Handling

- **Model not found error**: Run `python -m spacy download en_core_web_sm`.
- **Textatistic error**: If you have trouble installing `textatistic`, the bot will simply skip the advanced readability metrics but will still work.
- **Missing dataset**: If `bmw.csv` is missing, the app will show an error in the UI.


## Acknowledgments

- **Streamlit**: For the web interface.
- **spaCy**: For robust NLP tasks.
- **TextBlob**: For sentiment analysis.



