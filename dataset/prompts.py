"""
Benchmark prompt dataset - 9 top-level categories, 12 prompts each (108 total).
Each prompt has a unique id so benchmark results can be tracked/resumed reliably.
"""

PROMPTS = [

    # ---------------- 1. KNOWLEDGE ----------------
    {"id": "know_01", "category": "knowledge", "subcategory": "general_qa", "prompt": "What is RAM and why does a computer need it?"},
    {"id": "know_02", "category": "knowledge", "subcategory": "general_qa", "prompt": "What is the difference between HTTP and HTTPS?"},
    {"id": "know_03", "category": "knowledge", "subcategory": "explanation", "prompt": "Explain how a transistor works in simple terms."},
    {"id": "know_04", "category": "knowledge", "subcategory": "explanation", "prompt": "Explain the concept of recursion using a real-world analogy."},
    {"id": "know_05", "category": "knowledge", "subcategory": "technical", "prompt": "What is the difference between TCP and UDP?"},
    {"id": "know_06", "category": "knowledge", "subcategory": "technical", "prompt": "Explain what an operating system's kernel does."},
    {"id": "know_07", "category": "knowledge", "subcategory": "technical", "prompt": "What is the difference between a compiler and an interpreter?"},
    {"id": "know_08", "category": "knowledge", "subcategory": "science", "prompt": "Explain how photosynthesis works."},
    {"id": "know_09", "category": "knowledge", "subcategory": "science", "prompt": "Why does ice float on water?"},
    {"id": "know_10", "category": "knowledge", "subcategory": "science", "prompt": "What causes lightning during a thunderstorm?"},
    {"id": "know_11", "category": "knowledge", "subcategory": "general_qa", "prompt": "What is cloud computing?"},
    {"id": "know_12", "category": "knowledge", "subcategory": "explanation", "prompt": "Explain how a search engine works."},

    # ---------------- 2. REASONING ----------------
    {"id": "reas_01", "category": "reasoning", "subcategory": "reasoning", "prompt": "If all cats are animals, and some animals are pets, can we conclude all cats are pets? Explain your reasoning."},
    {"id": "reas_02", "category": "reasoning", "subcategory": "reasoning", "prompt": "A train leaves at 3 PM and travels at 60 km/h. Another leaves at 4 PM at 90 km/h from the same point, same direction. When does the second train catch up?"},
    {"id": "reas_03", "category": "reasoning", "subcategory": "mathematics", "prompt": "Solve for x: 3x + 7 = 22"},
    {"id": "reas_04", "category": "reasoning", "subcategory": "mathematics", "prompt": "What is the derivative of f(x) = x^3 + 2x^2 - 5x + 1?"},
    {"id": "reas_05", "category": "reasoning", "subcategory": "mathematics", "prompt": "Find the area of a circle with radius 7 cm."},
    {"id": "reas_06", "category": "reasoning", "subcategory": "math_word_problem", "prompt": "A shopkeeper buys apples at Rs. 20 per kg and sells at Rs. 28 per kg. What is the profit percentage?"},
    {"id": "reas_07", "category": "reasoning", "subcategory": "math_word_problem", "prompt": "If 5 workers can build a wall in 12 days, how many days will 3 workers take?"},
    {"id": "reas_08", "category": "reasoning", "subcategory": "decision_support", "prompt": "I have to choose between two job offers: one with higher salary but longer hours, another with lower salary but better work-life balance. What factors should I consider?"},
    {"id": "reas_09", "category": "reasoning", "subcategory": "decision_support", "prompt": "Should a startup prioritize hiring more engineers or spending on marketing first? Explain the trade-offs."},
    {"id": "reas_10", "category": "reasoning", "subcategory": "reasoning", "prompt": "Compare quicksort and mergesort in terms of time complexity and practical performance."},
    {"id": "reas_11", "category": "reasoning", "subcategory": "mathematics", "prompt": "Prove that the sum of two even numbers is always even."},
    {"id": "reas_12", "category": "reasoning", "subcategory": "reasoning", "prompt": "Explain why a hash table has average O(1) lookup time but worst case O(n)."},

    # ---------------- 3. PROGRAMMING ----------------
    {"id": "prog_01", "category": "programming", "subcategory": "coding", "prompt": "Write a Python function to check if a number is prime."},
    {"id": "prog_02", "category": "programming", "subcategory": "coding", "prompt": "Write a Python function to reverse a linked list."},
    {"id": "prog_03", "category": "programming", "subcategory": "coding", "prompt": "Write a C++ implementation of binary search."},
    {"id": "prog_04", "category": "programming", "subcategory": "coding", "prompt": "Write a Python script to read a CSV file and print the average of a numeric column."},
    {"id": "prog_05", "category": "programming", "subcategory": "coding", "prompt": "Implement a simple LRU cache in Python."},
    {"id": "prog_06", "category": "programming", "subcategory": "coding", "prompt": "Write a SQL query to find the second highest salary from an Employee table."},
    {"id": "prog_07", "category": "programming", "subcategory": "code_debugging", "prompt": "Find the bug in this code:\ndef factorial(n):\n    if n == 0:\n        return 0\n    return n * factorial(n-1)"},
    {"id": "prog_08", "category": "programming", "subcategory": "code_debugging", "prompt": "This Python code throws an IndexError. Fix it:\narr = [1,2,3]\nfor i in range(4):\n    print(arr[i])"},
    {"id": "prog_09", "category": "programming", "subcategory": "code_debugging", "prompt": "Why does this loop never terminate?\ni = 0\nwhile i < 10:\n    print(i)\n    i = i - 1"},
    {"id": "prog_10", "category": "programming", "subcategory": "coding", "prompt": "Write a multithreaded producer-consumer example in Python using threading and queue."},
    {"id": "prog_11", "category": "programming", "subcategory": "coding", "prompt": "Write a regular expression to validate an email address."},
    {"id": "prog_12", "category": "programming", "subcategory": "code_debugging", "prompt": "Explain why this recursive Fibonacci function is slow for large n, and how to optimize it:\ndef fib(n):\n    if n <= 1: return n\n    return fib(n-1) + fib(n-2)"},

    # ---------------- 4. LANGUAGE ----------------
    {"id": "lang_01", "category": "language", "subcategory": "translation", "prompt": "Translate this sentence to Hindi: 'The weather is very pleasant today.'"},
    {"id": "lang_02", "category": "language", "subcategory": "translation", "prompt": "Translate this sentence to French: 'I would like to book a table for two.'"},
    {"id": "lang_03", "category": "language", "subcategory": "rewriting", "prompt": "Rewrite this sentence to sound more formal: 'Hey, can you send me that file ASAP?'"},
    {"id": "lang_04", "category": "language", "subcategory": "rewriting", "prompt": "Rewrite this paragraph to be more concise: 'In today's modern world of today, it is very important and crucial for students to learn and understand the basics of programming.'"},
    {"id": "lang_05", "category": "language", "subcategory": "multilingual", "prompt": "What does 'Bonjour, comment ça va?' mean in English?"},
    {"id": "lang_06", "category": "language", "subcategory": "multilingual", "prompt": "How do you say 'Thank you very much' in Japanese?"},
    {"id": "lang_07", "category": "language", "subcategory": "email_writing", "prompt": "Write a polite email to a professor asking for a one-day extension on an assignment deadline."},
    {"id": "lang_08", "category": "language", "subcategory": "email_writing", "prompt": "Write a follow-up email after a job interview thanking the interviewer."},
    {"id": "lang_09", "category": "language", "subcategory": "rewriting", "prompt": "Fix the grammar in this sentence: 'He don't know nothing about the project deadline.'"},
    {"id": "lang_10", "category": "language", "subcategory": "translation", "prompt": "Translate this to Spanish: 'Where is the nearest train station?'"},
    {"id": "lang_11", "category": "language", "subcategory": "email_writing", "prompt": "Write a short email to a supervisor requesting approval for a new laptop for a project."},
    {"id": "lang_12", "category": "language", "subcategory": "rewriting", "prompt": "Simplify this sentence for a non-technical audience: 'The algorithm exhibits logarithmic time complexity due to its divide-and-conquer approach.'"},

    # ---------------- 5. CONTENT ----------------
    {"id": "cont_01", "category": "content", "subcategory": "summarization", "prompt": "Summarize this paragraph in 50 words: 'Artificial intelligence has rapidly transformed multiple industries over the past decade. From healthcare diagnostics to autonomous vehicles, AI systems are increasingly capable of performing tasks that once required human expertise. However, this progress raises important questions about ethics, employment, and the need for regulation to ensure responsible deployment.'"},
    {"id": "cont_02", "category": "content", "subcategory": "summarization", "prompt": "Summarize the plot of a typical hero's journey story in 3 sentences."},
    {"id": "cont_03", "category": "content", "subcategory": "creative_writing", "prompt": "Write a short 100-word story about a robot who discovers emotions."},
    {"id": "cont_04", "category": "content", "subcategory": "creative_writing", "prompt": "Write a short poem about monsoon rain in India."},
    {"id": "cont_05", "category": "content", "subcategory": "creative_ideation", "prompt": "Give me 5 creative name ideas for a college tech fest."},
    {"id": "cont_06", "category": "content", "subcategory": "creative_ideation", "prompt": "Suggest 5 unique project ideas for a final-year computer science student."},
    {"id": "cont_07", "category": "content", "subcategory": "summarization", "prompt": "Summarize the key differences between supervised and unsupervised learning in 3 bullet points."},
    {"id": "cont_08", "category": "content", "subcategory": "creative_writing", "prompt": "Write a two-line tagline for a mobile app that helps students manage study time."},
    {"id": "cont_09", "category": "content", "subcategory": "creative_ideation", "prompt": "Brainstorm 5 features a smart local AI assistant app could have."},
    {"id": "cont_10", "category": "content", "subcategory": "summarization", "prompt": "Summarize what a B.Tech final year project typically involves, in 3 sentences."},
    {"id": "cont_11", "category": "content", "subcategory": "creative_writing", "prompt": "Write a short motivational message for a student stressed about their final year project."},
    {"id": "cont_12", "category": "content", "subcategory": "creative_ideation", "prompt": "Suggest 3 catchy titles for a research paper about local AI model routing."},

    # ---------------- 6. DATA ----------------
    {"id": "data_01", "category": "data", "subcategory": "data_analysis", "prompt": "Given these sales figures: Jan-100, Feb-150, Mar-90, Apr-200, what trend do you observe and what might explain the drop in March?"},
    {"id": "data_02", "category": "data", "subcategory": "data_analysis", "prompt": "A dataset has 1000 rows and 15% missing values in one column. Suggest 3 ways to handle the missing data."},
    {"id": "data_03", "category": "data", "subcategory": "information_extraction", "prompt": "Extract the name, date, and amount from this text: 'Invoice for Mr. Ramesh Kumar dated 12 March 2024, total amount Rs. 4500.'"},
    {"id": "data_04", "category": "data", "subcategory": "information_extraction", "prompt": "Extract all email addresses from this text: 'Contact us at support@example.com or sales@example.org for queries.'"},
    {"id": "data_05", "category": "data", "subcategory": "classification", "prompt": "Classify this review as positive, negative, or neutral: 'The product was okay, nothing special but did the job.'"},
    {"id": "data_06", "category": "data", "subcategory": "classification", "prompt": "Classify this email as spam or not spam: 'Congratulations! You have won a $1000 gift card, click here to claim now!'"},
    {"id": "data_07", "category": "data", "subcategory": "data_analysis", "prompt": "What statistical measure would you use to detect outliers in a dataset, and why?"},
    {"id": "data_08", "category": "data", "subcategory": "information_extraction", "prompt": "Extract the key entities (person, organization, location) from: 'Elon Musk, CEO of Tesla, visited Berlin last week.'"},
    {"id": "data_09", "category": "data", "subcategory": "classification", "prompt": "Classify this support ticket by urgency (low/medium/high): 'The entire production server is down and customers cannot log in.'"},
    {"id": "data_10", "category": "data", "subcategory": "data_analysis", "prompt": "Explain the difference between correlation and causation with a simple example."},
    {"id": "data_11", "category": "data", "subcategory": "classification", "prompt": "Classify this text by topic (sports/technology/politics/health): 'The new smartphone features a faster processor and improved battery life.'"},
    {"id": "data_12", "category": "data", "subcategory": "information_extraction", "prompt": "Extract the main keywords from this sentence: 'Adaptive routing improves latency and reduces memory usage in local AI systems.'"},

    # ---------------- 7. TASK / PLANNING ----------------
    {"id": "task_01", "category": "task_planning", "subcategory": "planning", "prompt": "Create a 4-week study plan for learning the basics of machine learning from scratch."},
    {"id": "task_02", "category": "task_planning", "subcategory": "planning", "prompt": "Plan a day-by-day schedule for completing a B.Tech final year project in 3 months."},
    {"id": "task_03", "category": "task_planning", "subcategory": "instructional", "prompt": "Give step-by-step instructions for setting up a Python virtual environment on Windows."},
    {"id": "task_04", "category": "task_planning", "subcategory": "instructional", "prompt": "Explain step by step how to create a pull request on GitHub."},
    {"id": "task_05", "category": "task_planning", "subcategory": "document_qa", "prompt": "Based on this text: 'The meeting is scheduled for Monday at 10 AM in Room 204.' — what room is the meeting in?"},
    {"id": "task_06", "category": "task_planning", "subcategory": "document_qa", "prompt": "Based on this policy text: 'Employees must submit leave requests at least 3 days in advance.' — how many days notice is required?"},
    {"id": "task_07", "category": "task_planning", "subcategory": "planning", "prompt": "Outline a plan to organize a college tech fest with a budget of Rs. 50,000."},
    {"id": "task_08", "category": "task_planning", "subcategory": "instructional", "prompt": "Give step-by-step instructions to install and run a model using Ollama."},
    {"id": "task_09", "category": "task_planning", "subcategory": "planning", "prompt": "Create a weekly workout plan for a beginner focused on general fitness."},
    {"id": "task_10", "category": "task_planning", "subcategory": "document_qa", "prompt": "Based on this text: 'The warranty covers manufacturing defects for 12 months from the date of purchase.' — how long is the warranty period?"},
    {"id": "task_11", "category": "task_planning", "subcategory": "instructional", "prompt": "Explain step by step how to reset a forgotten Windows password."},
    {"id": "task_12", "category": "task_planning", "subcategory": "planning", "prompt": "Plan an agenda for a 1-hour project review meeting with a supervisor."},

    # ---------------- 8. CONVERSATION ----------------
    {"id": "conv_01", "category": "conversation", "subcategory": "conversation", "prompt": "Hi, how are you doing today?"},
    {"id": "conv_02", "category": "conversation", "subcategory": "conversation", "prompt": "Can you recommend a good movie to watch this weekend?"},
    {"id": "conv_03", "category": "conversation", "subcategory": "conversation", "prompt": "I'm feeling a bit stressed about my exams, any tips?"},
    {"id": "conv_04", "category": "conversation", "subcategory": "conversation", "prompt": "What's a good way to start a conversation with someone new?"},
    {"id": "conv_05", "category": "conversation", "subcategory": "conversation", "prompt": "Tell me an interesting fact I probably don't know."},
    {"id": "conv_06", "category": "conversation", "subcategory": "conversation", "prompt": "What do you think makes a good friend?"},
    {"id": "conv_07", "category": "conversation", "subcategory": "conversation", "prompt": "Can you suggest a fun weekend activity for a group of college friends?"},
    {"id": "conv_08", "category": "conversation", "subcategory": "conversation", "prompt": "What's your opinion on whether AI will replace most jobs?"},
    {"id": "conv_09", "category": "conversation", "subcategory": "conversation", "prompt": "How was your day going so far?"},
    {"id": "conv_10", "category": "conversation", "subcategory": "conversation", "prompt": "Any tips for staying motivated during a long project?"},
    {"id": "conv_11", "category": "conversation", "subcategory": "conversation", "prompt": "What's a good icebreaker question for a first team meeting?"},
    {"id": "conv_12", "category": "conversation", "subcategory": "conversation", "prompt": "Do you think it's better to work early morning or late at night?"},

    # ---------------- 9. OTHER ----------------
    {"id": "other_01", "category": "other", "subcategory": "other", "prompt": "What's the capital of Australia?"},
    {"id": "other_02", "category": "other", "subcategory": "other", "prompt": "Convert 25 degrees Celsius to Fahrenheit."},
    {"id": "other_03", "category": "other", "subcategory": "other", "prompt": "What year did the first man land on the moon?"},
    {"id": "other_04", "category": "other", "subcategory": "other", "prompt": "How many continents are there on Earth?"},
    {"id": "other_05", "category": "other", "subcategory": "other", "prompt": "What is the chemical symbol for gold?"},
    {"id": "other_06", "category": "other", "subcategory": "other", "prompt": "Convert 5 kilometers to miles."},
    {"id": "other_07", "category": "other", "subcategory": "other", "prompt": "What is the tallest mountain in the world?"},
    {"id": "other_08", "category": "other", "subcategory": "other", "prompt": "How many players are there in a cricket team?"},
    {"id": "other_09", "category": "other", "subcategory": "other", "prompt": "What is the currency of Japan?"},
    {"id": "other_10", "category": "other", "subcategory": "other", "prompt": "Convert 100 USD to a rough estimate in INR (approximate is fine)."},
    {"id": "other_11", "category": "other", "subcategory": "other", "prompt": "What is the boiling point of water at sea level in Celsius?"},
    {"id": "other_12", "category": "other", "subcategory": "other", "prompt": "Name the longest river in the world."},
]

CATEGORIES = sorted(set(p["category"] for p in PROMPTS))

if __name__ == "__main__":
    print(f"Total prompts: {len(PROMPTS)}")
    for cat in CATEGORIES:
        count = len([p for p in PROMPTS if p["category"] == cat])
        print(f"  {cat}: {count} prompts")