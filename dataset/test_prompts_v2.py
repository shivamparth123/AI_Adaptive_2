"""
Held-out test set v2 - 45 NEW prompts (5 per category), written AFTER the hybrid
classifier was designed and never used to tune keywords or train any model.

Use this for the final accuracy numbers: the original 36-prompt set's failures have
already been seen while diagnosing the classifier, so it is no longer a clean test.
Prompts are deliberately messy (lowercase, typos, no punctuation), like real usage.

Limitation to state in the report: written by the project's author/assistant, labels
are the author's judgement (some are genuinely ambiguous, e.g. "== vs === in js"),
and n=45 means one prompt is ~2.2 accuracy points.
"""

TEST_PROMPTS_V2 = [
    # knowledge
    {"id": "v2_know_01", "category": "knowledge", "prompt": "how does a bloom filter work"},
    {"id": "v2_know_02", "category": "knowledge", "prompt": "whats the point of virtual memory"},
    {"id": "v2_know_03", "category": "knowledge", "prompt": "why do airplanes leave white trails in the sky"},
    {"id": "v2_know_04", "category": "knowledge", "prompt": "what happens when you type a url in the browser"},
    {"id": "v2_know_05", "category": "knowledge", "prompt": "tell me how vaccines train the immune system"},

    # reasoning
    {"id": "v2_reas_01", "category": "reasoning", "prompt": "if 3 pumps fill a tank in 12 hours how long will 5 pumps take"},
    {"id": "v2_reas_02", "category": "reasoning", "prompt": "is the sum of two odd numbers always even? prove it"},
    {"id": "v2_reas_03", "category": "reasoning", "prompt": "a train leaves at 3pm going 60 mph, another at 4pm going 80 mph, when does the second catch up"},
    {"id": "v2_reas_04", "category": "reasoning", "prompt": "what's 15% of 840 minus 12% of 500"},
    {"id": "v2_reas_05", "category": "reasoning", "prompt": "which is larger, 2^100 or 100^20?"},

    # programming
    {"id": "v2_prog_01", "category": "programming", "prompt": "getting 'list index out of range' in my python code, why"},
    {"id": "v2_prog_02", "category": "programming", "prompt": "write sql to find the second highest salary from employees"},
    {"id": "v2_prog_03", "category": "programming", "prompt": "how to center a div in css"},
    {"id": "v2_prog_04", "category": "programming", "prompt": "difference between == and === in js"},
    {"id": "v2_prog_05", "category": "programming", "prompt": "can you write a function that checks if a string is a palindrome"},

    # language
    {"id": "v2_lang_01", "category": "language", "prompt": "how do you say thank you very much in japanese"},
    {"id": "v2_lang_02", "category": "language", "prompt": "can you proofread this: 'their going to the market tomorow'"},
    {"id": "v2_lang_03", "category": "language", "prompt": "rewrite this sentence to be less rude: 'your report is a mess'"},
    {"id": "v2_lang_04", "category": "language", "prompt": "translate 'i miss you' to french"},
    {"id": "v2_lang_05", "category": "language", "prompt": "what does 'carpe diem' mean"},

    # content
    {"id": "v2_cont_01", "category": "content", "prompt": "write a limerick about a lazy programmer"},
    {"id": "v2_cont_02", "category": "content", "prompt": "give me 10 name ideas for a podcast about space"},
    {"id": "v2_cont_03", "category": "content", "prompt": "summarise the plot of romeo and juliet in three sentences"},
    {"id": "v2_cont_04", "category": "content", "prompt": "draft a short speech for my sister's wedding"},
    {"id": "v2_cont_05", "category": "content", "prompt": "write an instagram caption for a sunset beach photo"},

    # data
    {"id": "v2_data_01", "category": "data", "prompt": "pull all the phone numbers out of this paragraph"},
    {"id": "v2_data_02", "category": "data", "prompt": "is this review positive or negative: 'worst pizza ever, never again'"},
    {"id": "v2_data_03", "category": "data", "prompt": "classify these emails as urgent or not urgent"},
    {"id": "v2_data_04", "category": "data", "prompt": "turn this json into a csv table"},
    {"id": "v2_data_05", "category": "data", "prompt": "find the outliers in 3, 5, 4, 6, 98, 5, 4"},

    # task_planning
    {"id": "v2_task_01", "category": "task_planning", "prompt": "make me a weekly meal prep plan for a vegetarian"},
    {"id": "v2_task_02", "category": "task_planning", "prompt": "how do i set up a python virtual environment"},
    {"id": "v2_task_03", "category": "task_planning", "prompt": "give me a packing checklist for a 3 day trek"},
    {"id": "v2_task_04", "category": "task_planning", "prompt": "plan my study timetable for the 2 weeks before finals"},
    {"id": "v2_task_05", "category": "task_planning", "prompt": "what are the steps to deploy a flask app on heroku"},

    # conversation
    {"id": "v2_conv_01", "category": "conversation", "prompt": "ugh mondays, am i right"},
    {"id": "v2_conv_02", "category": "conversation", "prompt": "do you like music?"},
    {"id": "v2_conv_03", "category": "conversation", "prompt": "i had a terrible day at work"},
    {"id": "v2_conv_04", "category": "conversation", "prompt": "good morning! how's it going"},
    {"id": "v2_conv_05", "category": "conversation", "prompt": "can we just chat for a bit"},

    # other (quick factual lookups / conversions)
    {"id": "v2_other_01", "category": "other", "prompt": "what is the capital of canada"},
    {"id": "v2_other_02", "category": "other", "prompt": "convert 5 miles to kilometres"},
    {"id": "v2_other_03", "category": "other", "prompt": "how many days are in a leap year"},
    {"id": "v2_other_04", "category": "other", "prompt": "who painted the mona lisa"},
    {"id": "v2_other_05", "category": "other", "prompt": "what's the boiling point of water in fahrenheit"},
]