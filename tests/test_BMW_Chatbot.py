import pytest
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from BMW_Chatbot import BMWChatbot

@pytest.fixture(scope="module")
def bot():
    csv_path = os.path.join("data", "bmw.csv")
    return BMWChatbot(csv_path)

qna_examples = [
    # General Information / Car Details
    "Tell me about the 2017 BMW 3 Series diesel.",
    "What is the price of a 2018 BMW X5?",
    "Can you give me details about the 2020 BMW 5 Series petrol?",
    "How much mileage does the 2019 BMW 3 Series get?",
    "What’s the fuel type of the 2021 BMW M3?",
    "Tell me about the 2022 BMW X3 with automatic transmission.",
    "What engine size does the 2020 BMW 7 Series have?",
    "How much tax is required for a 2021 BMW 4 Series?",
    "What is the transmission type for the 2018 BMW M4?",
    "Can you tell me about the fuel efficiency of the 2017 BMW 2 Series?",
    # Price and Cost-Related Questions
    "What is the price of a 2020 BMW 3 Series?",
    "How much is a 2019 BMW X6 in the UK?",
    "What’s the most expensive BMW available in 2021?",
    "Can you give me the average price of the BMW 5 Series 2020?",
    "How much does a 2022 BMW 4 Series cost in the market?",
    # Fuel and Transmission-Related Questions
    "What type of fuel does the 2019 BMW 3 Series use?",
    "Is the 2020 BMW X5 hybrid or petrol?",
    "Does the 2021 BMW 4 Series come in diesel?",
    "What transmission does the 2018 BMW 7 Series have?",
    "Is the 2020 BMW M4 automatic or manual?",
    # Year and Model Specific Questions
    "Tell me about the 2016 BMW 5 Series.",
    "What are the details of the 2022 BMW 3 Series?",
    "What year was the BMW 3 Series with the lowest price?",
    "How does the 2020 BMW M5 compare to the 2021 model?",
    "Is there any difference between the 2018 and 2020 BMW X1?",
    # Performance and Mileage Questions
    "How many miles does the 2017 BMW 3 Series have?",
    "What is the mileage of the 2020 BMW 5 Series diesel?",
    "How fuel-efficient is the 2021 BMW M3?",
    "How many miles can the 2019 BMW X5 drive on a full tank?",
    "What’s the top speed of the 2021 BMW 4 Series?",
    # Car Series-Specific Questions
    "What’s the difference between the 3 Series and 4 Series?",
    "How does the 1 Series compare to the 2 Series in terms of fuel economy?",
    "Tell me about the BMW 5 Series vs the BMW 7 Series.",
    "What makes the BMW X3 different from the BMW X5?",
    "What are the features of the 2020 BMW 6 Series?",
    # Aggregate and Price Comparison Questions
    "What’s the cheapest 2019 BMW 3 Series available?",
    "Tell me the average price of the 2018 BMW 5 Series.",
    "Which is the most expensive BMW 7 Series in 2021?",
    "What’s the least expensive BMW X1 available in 2020?",
    "What’s the highest price for a 2019 BMW X6?",
    # Miscellaneous / Specific Details
    "What is the engine size of the 2020 BMW M5?",
    "Does the 2021 BMW X7 come with a hybrid engine?",
    "What color options are available for the 2019 BMW 3 Series?",
    "Does the 2017 BMW 3 Series have a sunroof?",
    "What’s the safety rating of the 2021 BMW M4?",
    # Sentiment/Review Analysis Questions
    "Can you analyze this review: 'This BMW 3 Series is excellent. It has great mileage but is a bit expensive.'",
    "What is the sentiment of the review: 'The BMW 7 Series is fantastic but has high running costs.'",
    "Analyze the following review: 'The BMW X5 is an excellent family car, but its fuel economy could be better.'",
    "What is the sentiment of: 'I love my 2020 BMW 5 Series! It’s fast and luxurious!'",
    "Can you tell me the readability of this review: 'The 2018 BMW 3 Series is fast, has great comfort features, and performs well on highways.'",
    # Budget & Recommendation Questions
    "My budget is 15000.",
    "Recommend a car under 20k.",
    "I have a budget of $12,000 for a 1 Series.",
    "Show me cars less than 18000.",
    "Which car is best?",
    "Recommend me a BMW.",
    "What do you suggest for a 5 Series?",
    "Best car under 25k.",
    "Suggest a cheap diesel car.",
    "What is the best 3 Series you have?",
]

@pytest.mark.parametrize("query", qna_examples)
def test_qna_examples(bot, query):
    reply = bot.generate_reply(query)
    assert reply and isinstance(reply, str) and len(reply.strip()) > 0
