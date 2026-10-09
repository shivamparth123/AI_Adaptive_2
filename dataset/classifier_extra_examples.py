"""
Supplementary classifier training examples.

These exist ONLY to give the classifiers more training data. They are NOT part of
prompts.py, are NOT run through the LLM benchmark, and are NOT the held-out test set
(dataset/test_prompts.py). Written independently of test_prompts.py to avoid
indirectly tuning to it - general phrasing variety, not copies of test phrasings.
"""

EXTRA_TRAINING_EXAMPLES = [
    # knowledge
    {"id": "x_know_01", "category": "knowledge", "prompt": "why do metals conduct electricity better than plastic"},
    {"id": "x_know_02", "category": "knowledge", "prompt": "what causes jet lag"},
    {"id": "x_know_03", "category": "knowledge", "prompt": "can you tell me how vaccines work"},

    # reasoning
    {"id": "x_reas_01", "category": "reasoning", "prompt": "two trains 300km apart move toward each other at 60 and 90 km/h, when do they meet"},
    {"id": "x_reas_02", "category": "reasoning", "prompt": "is this argument valid: all squares are rectangles, this shape is a rectangle, so it must be a square"},
    {"id": "x_reas_03", "category": "reasoning", "prompt": "which is bigger asymptotically, 2^n or n^100"},

    # programming
    {"id": "x_prog_01", "category": "programming", "prompt": "my for loop skips the last element, can you fix it"},
    {"id": "x_prog_02", "category": "programming", "prompt": "convert this list of strings to integers in python"},
    {"id": "x_prog_03", "category": "programming", "prompt": "write a function that merges two sorted lists"},

    # language
    {"id": "x_lang_01", "category": "language", "prompt": "how do i politely decline a wedding invite"},
    {"id": "x_lang_02", "category": "language", "prompt": "what's a formal way to say 'I don't know'"},
    {"id": "x_lang_03", "category": "language", "prompt": "how do you say excuse me in korean"},

    # content
    {"id": "x_cont_01", "category": "content", "prompt": "write a caption for a sunset beach photo"},
    {"id": "x_cont_02", "category": "content", "prompt": "give me name ideas for a fitness app"},
    {"id": "x_cont_03", "category": "content", "prompt": "write a haiku about winter mornings"},

    # data
    {"id": "x_data_01", "category": "data", "prompt": "pull the phone number and company name out of this text"},
    {"id": "x_data_02", "category": "data", "prompt": "does this comment sound sarcastic or genuine"},
    {"id": "x_data_03", "category": "data", "prompt": "is this transaction description an expense or income"},

    # task_planning
    {"id": "x_task_01", "category": "task_planning", "prompt": "give me a checklist before publishing a mobile app"},
    {"id": "x_task_02", "category": "task_planning", "prompt": "walk me through resetting my router"},
    {"id": "x_task_03", "category": "task_planning", "prompt": "what's the process to renew a passport"},

    # conversation
    {"id": "x_conv_01", "category": "conversation", "prompt": "hi there, how's it going"},
    {"id": "x_conv_02", "category": "conversation", "prompt": "I can't decide what to eat tonight, any ideas"},
    {"id": "x_conv_03", "category": "conversation", "prompt": "just got some good news, feeling great today"},

    # other
    {"id": "x_other_01", "category": "other", "prompt": "how many bones are in the human body"},
    {"id": "x_other_02", "category": "other", "prompt": "convert 10 miles to kilometers"},
    {"id": "x_other_03", "category": "other", "prompt": "who painted the Mona Lisa"},
]