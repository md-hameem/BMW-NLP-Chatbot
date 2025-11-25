"""
BMWBot: A BMW Car Information Chatbot Using Classical NLP
----------------------------------------------------------

Features:
- Answers questions about BMW cars from bmw.csv using TF-IDF + cosine similarity.
- Handles greetings, thanks, help, and exit intents.
- Uses simple NLP to detect year and fuel type and filters the dataset accordingly.
- Can answer aggregate questions:
    - "average price of 3 series 2017"
    - "cheapest 5 series diesel"
    - "most expensive 1 series"
- Logs all interactions to chat_log.txt
- Has a special mode to analyze sentiment + readability of a text:
    - "analyze: <your review text>"
"""

import os




import re
import sys
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import pandas as pd
import numpy as np

import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from textblob import TextBlob

try:
    from textatistic import Textatistic
    HAS_TEXTATISTIC = True
except ImportError:
    HAS_TEXTATISTIC = False


# Data Classes

@dataclass
class QAPair:
    """Represents a single knowledge item derived from the BMW dataset."""
    question_text: str  
    answer_text: str     
    row_index: int       


def basic_clean(text: str) -> str:
    """Lowercase and remove non-alphanumeric characters except spaces."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_years(text: str) -> List[int]:
    """Extract plausible car years (e.g., 1990–2025) from text."""
    years = re.findall(r"\b(19[5-9]\d|20[0-2]\d|2025)\b", text)
    return [int(y) for y in years]


def extract_series(text: str) -> Optional[str]:
    """
    Extract BMW 'Series' from text, e.g. '1 Series', '3 series', etc.
    Returns something like '1 Series' or None.
    """
    m = re.search(r"\b([1-8])\s*series\b", text, flags=re.IGNORECASE)
    if m:
        num = m.group(1)
        return f"{num} Series"
    return None


def extract_fuel_keywords(text: str) -> Optional[str]:
    """Detect fuel type words in the query (diesel, petrol, hybrid)."""
    t = text.lower()
    if "diesel" in t:
        return "Diesel"
    if "petrol" in t or "gasoline" in t:
        return "Petrol"
    if "hybrid" in t:
        return "Hybrid"
    if "electric" in t:
        return "Electric"
    return None


class BMWChatbot:
    def __init__(self, csv_path: str):
        self.debug: bool = False

        self.nlp = spacy.load("en_core_web_sm")

        self.df = self._load_dataset(csv_path)

        self.qa_pairs: List[QAPair] = self._build_qa_pairs(self.df)

        self.vectorizer, self.qa_matrix = self._build_vectorizer(self.qa_pairs)

        self.corpus_clean = [basic_clean(q.question_text) for q in self.qa_pairs]


    def _load_dataset(self, csv_path: str) -> pd.DataFrame:
        """
        Load BMW dataset from CSV and perform minimal cleaning.
        """
        try:
            df = pd.read_csv(csv_path)
        except FileNotFoundError:
            print(f"[ERROR] Could not find file: {csv_path}")
            sys.exit(1)

        expected_cols = [
            "model", "year", "price", "transmission",
            "mileage", "fuelType", "tax", "mpg", "engineSize"
        ]
        missing = [c for c in expected_cols if c not in df.columns]
        if missing:
            print("[WARNING] Missing columns in CSV:", missing)

        # Drop rows where model or year are missing (simple cleaning)
        if "model" in df.columns and "year" in df.columns:
            df = df.dropna(subset=["model", "year"])
        df = df.reset_index(drop=True)
        return df


    def _row_to_question_text(self, row: pd.Series) -> str:
        """
        Create a synthetic 'question/description' string for each row.
        This is used for similarity matching with user queries.
        """
        model = str(row.get("model", ""))
        year = str(row.get("year", ""))
        fuel = str(row.get("fuelType", ""))
        transmission = str(row.get("transmission", ""))
        engine_size = str(row.get("engineSize", ""))
        mileage = str(row.get("mileage", ""))

        text = f"{year} bmw {model} {fuel} {transmission} {engine_size} litre {mileage} miles"
        return text

    def _row_to_answer_text(self, row: pd.Series) -> str:
        """
        Create a readable answer text for each car row.
        This is what the user will see.
        """
        model = row.get("model", "Unknown model")
        year = row.get("year", "Unknown year")
        price = row.get("price", "Unknown price")
        transmission = row.get("transmission", "Unknown transmission")
        mileage = row.get("mileage", "Unknown mileage")
        fuel = row.get("fuelType", "Unknown fuel")
        tax = row.get("tax", "Unknown tax")
        mpg = row.get("mpg", "Unknown mpg")
        engine_size = row.get("engineSize", "Unknown engine size")

        answer = (
            f"This is a {year} BMW {model}.\n"
            f"- Price: ${price}\n"
            f"- Transmission: {transmission}\n"
            f"- Fuel type: {fuel}\n"
            f"- Engine size: {engine_size} L\n"
            f"- Mileage: {mileage} miles\n"
            f"- Tax: ${tax} per year\n"
            f"- Fuel efficiency: {mpg} mpg\n"
        )
        return answer

    def _build_qa_pairs(self, df: pd.DataFrame) -> List[QAPair]:
        """Convert each row into a QAPair for matching."""
        qa_pairs: List[QAPair] = []
        for idx, row in df.iterrows():
            q_text = self._row_to_question_text(row)
            a_text = self._row_to_answer_text(row)
            qa_pairs.append(QAPair(question_text=q_text, answer_text=a_text, row_index=idx))
        return qa_pairs


    def _build_vectorizer(
        self, qa_pairs: List[QAPair]
    ) -> Tuple[TfidfVectorizer, np.ndarray]:
        """Build TF-IDF vectorizer on the question_text of all QAPairs."""
        corpus = [basic_clean(q.question_text) for q in qa_pairs]
        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform(corpus)
        return vectorizer, matrix


    def _analyze_query(self, text: str) -> Dict:
        """
        Perform some light NLP on the user query:
        - Extract years
        - Extract possible BMW series
        - Extract fuel keywords
        - Extract nouns (not heavily used yet, but good for extension)
        """
        cleaned = basic_clean(text)
        doc = self.nlp(cleaned)

        years = extract_years(text)
        series = extract_series(text)
        fuel = extract_fuel_keywords(text)
        noun_tokens = [token.text for token in doc if token.pos_ in ("NOUN", "PROPN")]

        return {
            "cleaned": cleaned,
            "years": years,
            "series": series,
            "fuel": fuel,
            "nouns": noun_tokens,
        }

    def _extract_price_from_query(self, text: str) -> Optional[float]:
        """
        Attempt to extract a budget/price value from the query.
        Handles: '20000', '20,000', '$20000', '20k'.
        """
        # Remove currency symbols and commas
        clean = text.lower().replace("$", "").replace(",", "").replace("£", "").replace("€", "")
        
        # Check for 'k' suffix (e.g., 20k)
        match_k = re.search(r"\b(\d+(\.\d+)?)k\b", clean)
        if match_k:
            return float(match_k.group(1)) * 1000
            
        # Find all numbers
        matches = re.findall(r"\b\d+\b", clean)
        if matches:
            # Heuristic: assume the budget is the largest number found 
            # (to avoid picking up '2017' year as budget if user says 'budget for 2017 car is 15000')
            numbers = [float(x) for x in matches]
            return max(numbers)
            
        return None

    # ---------------------------- Intent Logic -------------------------- #

    def _classify_intent(self, user_query: str) -> Dict:
        """
        Classify the intent of the user query into categories.
        Returns a dict with 'name' and optional extra info.
        """
        q = user_query.strip().lower()

        # Exit
        if q in {"exit", "quit", "bye"}:
            return {"name": "exit"}

        # Debug mode
        if q == "debug on":
            return {"name": "debug_on"}
        if q == "debug off":
            return {"name": "debug_off"}

        # Greeting
        greet_words = {"hi", "hello", "hey", "yo"}
        if any(q.startswith(w) for w in greet_words) or q in greet_words:
            return {"name": "greet"}

        # Thanks
        if "thank" in q or "thanks" in q:
            return {"name": "thanks"}

        # Help
        if "help" in q or "what can you do" in q:
            return {"name": "help"}

        # Readability / sentiment mode
        if q.startswith("analyze:") or "analyze this review" in q or "readability" in q:
            return {"name": "analyze_text"}

        # Aggregate questions
        if "average price" in q or "avg price" in q or "mean price" in q:
            return {"name": "aggregate_avg"}
        if "cheapest" in q or "lowest price" in q or "least expensive" in q:
            return {"name": "aggregate_cheapest"}
        if "most expensive" in q or "highest price" in q:
            return {"name": "aggregate_most_expensive"}

        # NEW: Budget / Recommendation with constraint
        # Matches: "my budget is...", "under 10000", "less than 15k"
        has_digit = any(c.isdigit() for c in q)
        if "budget" in q or (("under" in q or "less than" in q or "cheaper than" in q) and has_digit):
            return {"name": "recommend_budget"}

        # NEW: General Recommendation / "Best" car
        # Matches: "which car is best", "recommend me a car", "what do you suggest"
        if "best" in q or "recommend" in q or "suggest" in q or "which car" in q:
            return {"name": "recommend_best"}

        # Default: car info query
        return {"name": "car_info"}

    # ------------------------ Matching & Reply -------------------------- #

    def _filter_indices_by_analysis(self, analysis: Dict) -> List[int]:
        """
        Use extracted year / series / fuel to filter which QAPairs to consider.
        Returns list of row indices into self.qa_pairs.
        """
        years = analysis.get("years", [])
        series = analysis.get("series", None)
        fuel = analysis.get("fuel", None)

        df = self.df

        # Start with all indices
        mask = pd.Series([True] * len(df))

        if years:
            mask &= df["year"].isin(years)

        if series is not None and "model" in df.columns:
            mask &= df["model"].str.contains(series, case=False, na=False)

        if fuel is not None and "fuelType" in df.columns:
            mask &= df["fuelType"].str.contains(fuel, case=False, na=False)

        indices = list(df[mask].index)

        # If filtering removed everything, fall back to all indices
        if not indices:
            indices = list(df.index)

        if self.debug:
            print(f"[DEBUG] Filtered candidate indices: {indices[:10]} (showing up to 10)")

        return indices

    def _find_best_match(
        self,
        user_query: str,
        candidate_indices: Optional[List[int]] = None,
        min_similarity: float = 0.1,
    ) -> Optional[Tuple[QAPair, float, List[Tuple[QAPair, float]]]]:
        """
        Find the most similar QAPair to the user query using cosine similarity.
        Optionally restrict to a subset of candidate_indices.
        Returns:
            (best_QAPair, best_score, top3_list) or None if similarity is too low.
        top3_list = list of (QAPair, score) for top 3 matches for debugging/logging.
        """
        cleaned_query = basic_clean(user_query)
        if not cleaned_query:
            return None

        query_vec = self.vectorizer.transform([cleaned_query])

        if candidate_indices is None:
            sims = cosine_similarity(query_vec, self.qa_matrix)[0]
            full_indices = np.arange(len(self.qa_pairs))
        else:
            sub_matrix = self.qa_matrix[candidate_indices]
            sims = cosine_similarity(query_vec, sub_matrix)[0]
            full_indices = np.array(candidate_indices)

        best_sub_idx = int(np.argmax(sims))
        best_score = float(sims[best_sub_idx])
        best_global_idx = int(full_indices[best_sub_idx])

        if best_score < min_similarity:
            return None

        # Top 3 matches for debugging/logging
        top_k = min(3, len(sims))
        top_sub_indices = np.argsort(sims)[-top_k:][::-1]
        top3: List[Tuple[QAPair, float]] = []
        for sub_i in top_sub_indices:
            global_i = int(full_indices[sub_i])
            top3.append((self.qa_pairs[global_i], float(sims[sub_i])))

        if self.debug:
            print("[DEBUG] Top matches:")
            for pair, score in top3:
                print(f"  - {pair.question_text} (score={score:.3f})")

        return self.qa_pairs[best_global_idx], best_score, top3

    # --------------------- Aggregate Question Handlers ------------------ #

    def _filter_df_for_aggregate(self, user_query: str) -> pd.DataFrame:
        """
        Filter the dataframe according to user query (year, series, fuel).
        Used for aggregate price questions.
        """
        analysis = self._analyze_query(user_query)
        df = self.df.copy()

        years = analysis.get("years", [])
        series = analysis.get("series", None)
        fuel = analysis.get("fuel", None)

        if years:
            df = df[df["year"].isin(years)]
        if series:
            df = df[df["model"].str.contains(series, case=False, na=False)]
        if fuel:
            df = df[df["fuelType"].str.contains(fuel, case=False, na=False)]

        return df

    def _handle_aggregate_avg(self, user_query: str) -> str:
        df_filtered = self._filter_df_for_aggregate(user_query)
        if df_filtered.empty:
            return "I couldn't find matching cars to compute an average price."

        avg_price = df_filtered["price"].mean()
        count = len(df_filtered)
        return (
            f"Based on {count} matching BMW cars, the average price is approximately "
            f"${avg_price:,.0f}."
        )

    def _handle_aggregate_cheapest(self, user_query: str) -> str:
        df_filtered = self._filter_df_for_aggregate(user_query)
        if df_filtered.empty:
            return "I couldn't find matching cars to determine the cheapest one."

        idx_min = df_filtered["price"].idxmin()
        row = df_filtered.loc[idx_min]
        base_answer = self._row_to_answer_text(row)
        return "Here is the cheapest matching BMW:\n" + base_answer

    def _handle_aggregate_most_expensive(self, user_query: str) -> str:
        df_filtered = self._filter_df_for_aggregate(user_query)
        if df_filtered.empty:
            return "I couldn't find matching cars to determine the most expensive one."

        idx_max = df_filtered["price"].idxmax()
        row = df_filtered.loc[idx_max]
        base_answer = self._row_to_answer_text(row)
        return "Here is the most expensive matching BMW:\n" + base_answer

    # ----------------------- Recommendation Handlers -------------------- #

    def _handle_recommend_budget(self, user_query: str) -> str:
        """Handle queries like 'my budget is 15000'."""
        budget = self._extract_price_from_query(user_query)
        if not budget:
            return "I noticed you mentioned a budget, but I couldn't figure out the amount. Try saying 'My budget is 15000'."

        # Filter by budget
        df_filtered = self.df[self.df["price"] <= budget]
        
        # Also apply other filters if present (e.g. "budget is 15000 for a 3 series")
        analysis = self._analyze_query(user_query)
        if analysis.get("series"):
            df_filtered = df_filtered[df_filtered["model"].str.contains(analysis["series"], case=False, na=False)]
        
        if df_filtered.empty:
            return f"I couldn't find any BMW cars within a budget of ${budget:,.0f}."

        # Strategy: Recommend the newest car, then lowest mileage within budget
        df_sorted = df_filtered.sort_values(by=["year", "mileage"], ascending=[False, True])
        
        top_pick = df_sorted.iloc[0]
        count = len(df_filtered)
        
        reply = f"I found {count} cars within your budget of ${budget:,.0f}.\n"
        reply += "Here is the best option (newest with low mileage):\n"
        reply += self._row_to_answer_text(top_pick)
        
        return reply

    def _handle_recommend_best(self, user_query: str) -> str:
        """Handle queries like 'which car is best' or 'recommend me a car'."""
        # "Best" is subjective. We will offer:
        # 1. The Newest (Modern)
        # 2. The Most Efficient (MPG)
        
        # Apply filters if user said "best 3 series"
        df_filtered = self._filter_df_for_aggregate(user_query)
        
        if df_filtered.empty:
            return "I couldn't find any cars matching your criteria to make a recommendation."

        # 1. Newest
        idx_newest = df_filtered["year"].idxmax()
        row_newest = df_filtered.loc[idx_newest]
        
        # 2. Best MPG
        idx_mpg = df_filtered["mpg"].idxmax()
        row_mpg = df_filtered.loc[idx_mpg]
        
        reply = "Defining 'best' depends on what you need. Here are my top recommendations:\n\n"
        reply += "--- Option 1: The Newest Model ---\n"
        reply += self._row_to_answer_text(row_newest)
        reply += "\n"
        reply += "--- Option 2: Most Fuel Efficient (Highest MPG) ---\n"
        reply += self._row_to_answer_text(row_mpg)
        
        return reply

    # ----------------------- Text Analysis Handler ---------------------- #

    def _handle_text_analysis(self, user_query: str) -> str:
        """
        Analyze sentiment and readability of a provided text.
        Expected formats:
            - "analyze: <text here>"
            - "analyze this review: <text here>"
        """
        q = user_query.strip()
        lower = q.lower()

        if "analyze this review" in lower:
            parts = q.split(":", 1)
            text = parts[1].strip() if len(parts) > 1 else ""
        elif lower.startswith("analyze:"):
            text = q.split(":", 1)[1].strip()
        else:
            # Fallback: assume user pasted text right after "analyze"
            text = q.replace("analyze", "", 1).strip()

        if not text:
            return "Please provide the text to analyze, e.g. 'analyze: This BMW is very comfortable but a bit expensive.'"

        blob = TextBlob(text)
        polarity = blob.sentiment.polarity  # [-1, 1]
        subjectivity = blob.sentiment.subjectivity  # [0, 1]

        response_lines = []
        response_lines.append("Here is the analysis of your text:\n")
        response_lines.append(f"- Sentiment polarity (−1=negative, +1=positive): {polarity:.3f}")
        response_lines.append(f"- Subjectivity (0=objective, 1=subjective): {subjectivity:.3f}")

        if HAS_TEXTATISTIC:
            try:
                ta = Textatistic(text)
                stats = ta.dict()
                response_lines.append("\nReadability estimates:")
                # Show a few key metrics; adjust if needed
                for key in ["num_words", "num_sentences", "num_syllables"]:
                    if key in stats:
                        response_lines.append(f"- {key}: {stats[key]}")
                for key in ["flesch_score", "smog_index", "flesch_kincaid_grade"]:
                    if key in stats:
                        response_lines.append(f"- {key}: {stats[key]:.2f}")
            except Exception:
                response_lines.append("\n(Readability analysis failed unexpectedly.)")
        else:
            response_lines.append(
                "\n(Readability scores require the 'textatistic' package, "
                "which is not currently installed.)"
            )

        return "\n".join(response_lines)

    # -------------------------- Logging Helper -------------------------- #

    def _log_interaction(
        self,
        user_query: str,
        reply: str,
        top_matches: Optional[List[Tuple[QAPair, float]]] = None,
    ) -> None:
        """Append a simple log line to logs/chat_log.txt."""
        log_path = os.path.join("logs", "chat_log.txt")
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write("USER: " + user_query + "\n")
                f.write("BOT: " + reply + "\n")
                if top_matches:
                    f.write("TOP_MATCHES:\n")
                    for pair, score in top_matches:
                        f.write(f"  - {pair.question_text} (score={score:.3f})\n")
                f.write("-" * 40 + "\n")
        except Exception:
            # Logging should never crash the chatbot
            pass

    # -------------------------- Main Reply Logic ------------------------ #

    def _handle_greet(self) -> str:
        return (
            "Hi! I'm BMWBot, your BMW car information assistant.\n"
            "Ask me about BMW models, years, fuel types, prices, and more.\n"
            "Example: 'Tell me about 2017 3 Series diesel.'"
        )

    def _handle_help(self) -> str:
        return (
            "Here’s what I can do:\n"
            "- Car details: 'Tell me about 2017 3 Series diesel'\n"
            "- Recommendations: 'Which car is best?' or 'Recommend a 5 Series'\n"
            "- Budget help: 'My budget is 15000' or 'Cars under 20k'\n"
            "- Average price: 'average price of 3 Series 2017'\n"
            "- Cheapest car: 'cheapest 5 Series diesel'\n"
            "- Text analysis: 'analyze: This BMW is very comfortable.'\n"
            "- Toggle debug: 'debug on' or 'debug off'\n"
            "- Type 'exit' to quit."
        )

    def _handle_thanks(self) -> str:
        return "You're welcome! Let me know if you have more questions about BMW cars."

    def _handle_debug_on(self) -> str:
        self.debug = True
        return "Debug mode is now ON. I’ll show extra matching information."

    def _handle_debug_off(self) -> str:
        self.debug = False
        return "Debug mode is now OFF."

    def _handle_car_info(self, user_query: str) -> str:
        """Main car info handler using filtered TF-IDF matching."""
        analysis = self._analyze_query(user_query)
        candidate_indices = self._filter_indices_by_analysis(analysis)

        match = self._find_best_match(user_query, candidate_indices=candidate_indices)
        if match is None:
            return (
                "Sorry, I couldn't find a good match for your question.\n"
                "Try including the BMW model and year, e.g., 'Tell me about 2017 BMW 3 Series'."
            )

        qa_pair, score, top3 = match
        reply = qa_pair.answer_text
        reply += f"\n[Match confidence: {score:.2f}]"

        # Log with top matches for analysis
        self._log_interaction(user_query, reply, top_matches=top3)
        return reply

    def _is_valid_input(self, text: str) -> bool:
        """
        Analyze input to check if it looks like a valid question or command.
        Returns True if valid, False if likely gibberish or irrelevant.
        """
        t = text.strip().lower()
        if not t:
            return False
        
        # 1. Allow-list (Greetings, commands)
        allow_starts = {"hi", "hello", "hey", "yo", "help", "exit", "quit", "bye", "thank", "debug", "analyze"}
        if any(t.startswith(w) for w in allow_starts):
            return True

        # 2. Check for domain keywords
        keywords = {
            "bmw", "car", "vehicle", "series", "model", "price", "cost", "budget", 
            "year", "mileage", "fuel", "petrol", "diesel", "hybrid", "electric",
            "engine", "transmission", "auto", "manual", "tax", "mpg", "efficiency",
            "recommend", "suggest", "best", "cheap", "expensive", "average", "compare",
            "under", "less", "more", "most", "least"
        }
        tokens = set(re.findall(r"\w+", t))
        if not tokens.isdisjoint(keywords):
            return True

        # 3. Check for question words
        q_words = {"what", "which", "how", "can", "could", "would", "is", "are", "do", "does", "tell"}
        if not tokens.isdisjoint(q_words):
            return True

        # 4. Check for specific entities (Years, Prices)
        if extract_years(t):
            return True
        if self._extract_price_from_query(t):
            return True
            
        # 5. Check for specific series pattern like "x5", "m3", "z4"
        if re.search(r"\b[xX][1-7]\b", t) or re.search(r"\b[mM][2-8]\b", t) or re.search(r"\b[zZ]4\b", t) or re.search(r"\bi3\b", t) or re.search(r"\bi8\b", t):
            return True

        return False

    def generate_reply(self, user_query: str) -> str:
        """
        Main entry point: decide intent and route to appropriate handler.
        """
        # Validate input first
        if not self._is_valid_input(user_query):
            return "That doesn't look like a valid question. Please ask something about BMW cars (e.g., 'price of X5', 'best car under 20k')."

        intent = self._classify_intent(user_query)
        name = intent["name"]

        if name == "exit":
            reply = "Goodbye! Thanks for chatting with BMWBot."
            self._log_interaction(user_query, reply)
            return reply

        if name == "greet":
            reply = self._handle_greet()
            self._log_interaction(user_query, reply)
            return reply

        if name == "help":
            reply = self._handle_help()
            self._log_interaction(user_query, reply)
            return reply

        if name == "thanks":
            reply = self._handle_thanks()
            self._log_interaction(user_query, reply)
            return reply

        if name == "debug_on":
            reply = self._handle_debug_on()
            self._log_interaction(user_query, reply)
            return reply

        if name == "debug_off":
            reply = self._handle_debug_off()
            self._log_interaction(user_query, reply)
            return reply

        if name == "analyze_text":
            reply = self._handle_text_analysis(user_query)
            self._log_interaction(user_query, reply)
            return reply

        if name == "aggregate_avg":
            reply = self._handle_aggregate_avg(user_query)
            self._log_interaction(user_query, reply)
            return reply

        if name == "aggregate_cheapest":
            reply = self._handle_aggregate_cheapest(user_query)
            self._log_interaction(user_query, reply)
            return reply

        if name == "aggregate_most_expensive":
            reply = self._handle_aggregate_most_expensive(user_query)
            self._log_interaction(user_query, reply)
            return reply

        if name == "recommend_budget":
            reply = self._handle_recommend_budget(user_query)
            self._log_interaction(user_query, reply)
            return reply

        if name == "recommend_best":
            reply = self._handle_recommend_best(user_query)
            self._log_interaction(user_query, reply)
            return reply

        # Default: car info search
        reply = self._handle_car_info(user_query)
        # _handle_car_info already logs; no need to log again
        return reply



