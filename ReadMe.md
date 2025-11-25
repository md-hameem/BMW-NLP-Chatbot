# BMW Chatbot

## Overview
BMW Chatbot is an intelligent car information assistant powered by **Natural Language Processing (NLP)**. It helps users find details about BMW models, check prices, get recommendations based on budget, and analyze car reviews. The project now features a modern web interface built with **Streamlit**.

## Features
- **Car Information**: Fetches detailed specs (price, transmission, fuel, mileage, etc.) from a dataset.
- **Smart Recommendations**:
  - **Budget-Based**: "Find me a car under 15k" or "My budget is 20000".
  - **General**: "Which car is best?" (Suggests newest and most efficient models).
- **Aggregate Queries**: Answers questions like "average price of 3 Series" or "cheapest X5".
- **Input Validation**: Detects invalid inputs or gibberish and prompts for clarification.
- **Sentiment & Readability Analysis**: Analyzes user-provided text (e.g., reviews) for sentiment polarity and readability scores.
- **Web Interface**: Clean and interactive chat UI using Streamlit.

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
/bmw-chatbot
│
├── app.py                    # Streamlit web application entry point
├── BMW_Chatbot.py            # Core chatbot logic and NLP processing
├── bmw.csv                   # Dataset containing car information
├── requirements.txt          # Python dependencies
├── QNA_Example.md            # List of example questions for testing
└── README.md                 # Project documentation
```

## How It Works

1.  **NLP Processing**: The bot uses `spaCy` to extract entities like **Year**, **Series**, and **Fuel Type** from your query.
2.  **Filtering**: It filters the dataset (`bmw.csv`) to narrow down the search space.
3.  **Similarity Matching**: It uses **TF-IDF** and **Cosine Similarity** to find the most relevant car entry matching your description.
4.  **Intent Classification**: Rule-based logic determines if you are asking for a specific car, a recommendation, or an aggregate stat (avg/min/max).

## Customizing the Dataset

The `bmw.csv` file drives the chatbot's knowledge. It contains:
* `model`, `year`, `price`, `transmission`, `mileage`, `fuelType`, `tax`, `mpg`, `engineSize`

You can add new rows to this CSV to update the bot's knowledge base without changing the code.

## Troubleshooting

*   **Model not found error**: Run `python -m spacy download en_core_web_sm`.
*   **Textatistic error**: If you have trouble installing `textatistic`, the bot will simply skip the advanced readability metrics but will still work.

## Acknowledgments

*   **Streamlit**: For the web interface.
*   **spaCy**: For robust NLP tasks.
*   **TextBlob**: For sentiment analysis.


