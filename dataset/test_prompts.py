"""
Held-out test prompts for the query classifier comparison.

These 36 prompts (4 per category) were written AFTER the classifiers were built and
are NOT used for training the ML classifier or for writing the rule-based keywords.
They are used only to measure classification accuracy fairly.

Limitations to state in the report:
  - Written by the same author/assistant as the training prompts, so phrasing may
    still be similar in style. Adding your own messy, real-world prompts makes the
    test harder and more honest.
  - 36 items means each prompt is ~2.8 points of accuracy; small differences between
    classifiers are not statistically meaningful.
"""

TEST_PROMPTS = [
    # knowledge
    {"id": "t_know_01", "category": "knowledge", "prompt": "how do solid state drives store data without moving parts"},
    {"id": "t_know_02", "category": "knowledge", "prompt": "Why is the sky blue during the day but red at sunset?"},
    {"id": "t_know_03", "category": "knowledge", "prompt": "Can you explain what a firewall does?"},
    {"id": "t_know_04", "category": "knowledge", "prompt": "what's the difference between a virus and a bacterium"},

    # reasoning
    {"id": "t_reas_01", "category": "reasoning", "prompt": "A car covers 150 km in 2.5 hours. What's its average speed, and how long would 400 km take?"},
    {"id": "t_reas_02", "category": "reasoning", "prompt": "If a rectangle's perimeter is 36 and its length is twice its width, find its dimensions."},
    {"id": "t_reas_03", "category": "reasoning", "prompt": "Is it logically valid to say: no birds are mammals, bats are mammals, therefore bats are not birds?"},
    {"id": "t_reas_04", "category": "reasoning", "prompt": "Which grows faster, n log n or n squared, and why?"},

    # programming
    {"id": "t_prog_01", "category": "programming", "prompt": "my python loop keeps printing the same value forever, here's the code: while x > 0: print(x)"},
    {"id": "t_prog_02", "category": "programming", "prompt": "Write a JavaScript function that removes duplicates from an array"},
    {"id": "t_prog_03", "category": "programming", "prompt": "how do I reverse a string in C"},
    {"id": "t_prog_04", "category": "programming", "prompt": "Write a bash one-liner to count lines in all .txt files"},

    # language
    {"id": "t_lang_01", "category": "language", "prompt": "How would I say 'where is the library' in German?"},
    {"id": "t_lang_02", "category": "language", "prompt": "Make this sound more professional: 'we messed up the order, sorry'"},
    {"id": "t_lang_03", "category": "language", "prompt": "Translate 'good morning, how did you sleep?' into Spanish"},
    {"id": "t_lang_04", "category": "language", "prompt": "Write a short email declining a meeting invitation politely"},

    # content
    {"id": "t_cont_01", "category": "content", "prompt": "Give me a one-paragraph summary of the causes of World War 1"},
    {"id": "t_cont_02", "category": "content", "prompt": "Write a short poem about a rainy Monday morning"},
    {"id": "t_cont_03", "category": "content", "prompt": "Come up with 5 names for a coffee shop that also sells books"},
    {"id": "t_cont_04", "category": "content", "prompt": "Write a funny two-line caption for a photo of a cat on a laptop"},

    # data
    {"id": "t_data_01", "category": "data", "prompt": "From this sentence pull out the date and the person's name: 'Priya booked the hall on 14 August.'"},
    {"id": "t_data_02", "category": "data", "prompt": "Is this message spam? 'You've been selected for a free cruise, reply YES now'"},
    {"id": "t_data_03", "category": "data", "prompt": "Label the sentiment of this review: 'Battery life is awful but the screen is gorgeous.'"},
    {"id": "t_data_04", "category": "data", "prompt": "What does a high standard deviation tell you about a dataset?"},

    # task_planning
    {"id": "t_task_01", "category": "task_planning", "prompt": "Help me plan a 5-day revision schedule for my DBMS exam"},
    {"id": "t_task_02", "category": "task_planning", "prompt": "What are the steps to publish a Python package to PyPI?"},
    {"id": "t_task_03", "category": "task_planning", "prompt": "Create a checklist for moving into a new apartment"},
    {"id": "t_task_04", "category": "task_planning", "prompt": "Walk me through setting up SSH keys for GitHub"},

    # conversation
    {"id": "t_conv_01", "category": "conversation", "prompt": "hey! what's up?"},
    {"id": "t_conv_02", "category": "conversation", "prompt": "I'm bored, any suggestions for something to do tonight?"},
    {"id": "t_conv_03", "category": "conversation", "prompt": "Do you ever get tired of answering questions?"},
    {"id": "t_conv_04", "category": "conversation", "prompt": "I just finished my exams, feeling great!"},

    # other (quick factual lookups / conversions)
    {"id": "t_other_01", "category": "other", "prompt": "What is the speed of light in km/s?"},
    {"id": "t_other_02", "category": "other", "prompt": "Convert 72 inches to centimeters"},
    {"id": "t_other_03", "category": "other", "prompt": "Who wrote the play Hamlet?"},
    {"id": "t_other_04", "category": "other", "prompt": "What is the largest ocean on Earth?"},
]