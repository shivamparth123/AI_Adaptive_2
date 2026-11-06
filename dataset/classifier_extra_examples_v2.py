"""
Extra TRAINING examples (v2) for the ML classifier: 8 per category, written in the
terse, lowercase, student-style phrasing real users type (the original 108 benchmark
prompts are formal and well-punctuated, which is a poor match for real queries).

These are TRAINING data only. They were checked for near-duplicates against both held-out
test sets (test_prompts.py and test_prompts_v2.py) so they cannot leak into evaluation.
"""

EXTRA_TRAINING_EXAMPLES_V2 = [
    # knowledge
    {"id": "x2_know_01", "category": "knowledge", "prompt": "what is a semaphore in os"},
    {"id": "x2_know_02", "category": "knowledge", "prompt": "explain deadlock and how to avoid it"},
    {"id": "x2_know_03", "category": "knowledge", "prompt": "how do neural networks learn"},
    {"id": "x2_know_04", "category": "knowledge", "prompt": "difference between ac and dc current"},
    {"id": "x2_know_05", "category": "knowledge", "prompt": "why is the ocean salty"},
    {"id": "x2_know_06", "category": "knowledge", "prompt": "what is overfitting in machine learning"},
    {"id": "x2_know_07", "category": "knowledge", "prompt": "how does gps work"},
    {"id": "x2_know_08", "category": "knowledge", "prompt": "what is entropy in thermodynamics"},
    # reasoning
    {"id": "x2_reas_01", "category": "reasoning", "prompt": "find the eigenvalues of [[2,1],[1,2]]"},
    {"id": "x2_reas_02", "category": "reasoning", "prompt": "if x + y = 10 and x - y = 2 what is x"},
    {"id": "x2_reas_03", "category": "reasoning", "prompt": "probability of getting two heads in three coin tosses"},
    {"id": "x2_reas_04", "category": "reasoning", "prompt": "a car costs 8 lakh and loses 15% value every year, what is it worth after 3 years"},
    {"id": "x2_reas_05", "category": "reasoning", "prompt": "simplify (x^2 - 9)/(x - 3)"},
    {"id": "x2_reas_06", "category": "reasoning", "prompt": "is it better to rent or buy a house, compare the tradeoffs"},
    {"id": "x2_reas_07", "category": "reasoning", "prompt": "how many ways can 5 people sit in a row"},
    {"id": "x2_reas_08", "category": "reasoning", "prompt": "integrate x*e^x dx"},
    # programming
    {"id": "x2_prog_01", "category": "programming", "prompt": "my c program gives segmentation fault when i use pointers"},
    {"id": "x2_prog_02", "category": "programming", "prompt": "how to read a file line by line in java"},
    {"id": "x2_prog_03", "category": "programming", "prompt": "write a python program to count words in a text file"},
    {"id": "x2_prog_04", "category": "programming", "prompt": "what does this error mean: nullpointerexception"},
    {"id": "x2_prog_05", "category": "programming", "prompt": "implement bubble sort in c++"},
    {"id": "x2_prog_06", "category": "programming", "prompt": "git merge conflict how to resolve"},
    {"id": "x2_prog_07", "category": "programming", "prompt": "write a function to find factorial using recursion"},
    {"id": "x2_prog_08", "category": "programming", "prompt": "how to make a get request in javascript"},
    # language
    {"id": "x2_lang_01", "category": "language", "prompt": "translate good night to german"},
    {"id": "x2_lang_02", "category": "language", "prompt": "correct my english: i am having a doubt about this topic since yesterday"},
    {"id": "x2_lang_03", "category": "language", "prompt": "make this email sound more polite: send me the report today"},
    {"id": "x2_lang_04", "category": "language", "prompt": "what is the hindi word for water"},
    {"id": "x2_lang_05", "category": "language", "prompt": "paraphrase this line: the results were not as expected"},
    {"id": "x2_lang_06", "category": "language", "prompt": "write a leave application letter for two days"},
    {"id": "x2_lang_07", "category": "language", "prompt": "how to say how are you in spanish"},
    {"id": "x2_lang_08", "category": "language", "prompt": "whats the past tense of lead"},
    # content
    {"id": "x2_cont_01", "category": "content", "prompt": "write a short story about a dragon who is afraid of fire"},
    {"id": "x2_cont_02", "category": "content", "prompt": "give me ideas for a final year project on iot"},
    {"id": "x2_cont_03", "category": "content", "prompt": "write a haiku about autumn"},
    {"id": "x2_cont_04", "category": "content", "prompt": "suggest some catchy names for my youtube channel on coding"},
    {"id": "x2_cont_05", "category": "content", "prompt": "write a birthday message for my best friend"},
    {"id": "x2_cont_06", "category": "content", "prompt": "summarize the history of the internet in a paragraph"},
    {"id": "x2_cont_07", "category": "content", "prompt": "write a funny tweet about exams"},
    {"id": "x2_cont_08", "category": "content", "prompt": "come up with a slogan for a recycling campaign"},
    # data
    {"id": "x2_data_01", "category": "data", "prompt": "extract all dates from this text"},
    {"id": "x2_data_02", "category": "data", "prompt": "tag each sentence as question or statement"},
    {"id": "x2_data_03", "category": "data", "prompt": "which of these customer comments are complaints"},
    {"id": "x2_data_04", "category": "data", "prompt": "what is the mean and standard deviation of 12, 15, 11, 18, 14"},
    {"id": "x2_data_05", "category": "data", "prompt": "group these words into fruits and vegetables"},
    {"id": "x2_data_06", "category": "data", "prompt": "extract the names of all companies mentioned in this article"},
    {"id": "x2_data_07", "category": "data", "prompt": "label this tweet as sarcastic or sincere"},
    {"id": "x2_data_08", "category": "data", "prompt": "how to handle missing values in a dataset"},
    # task_planning
    {"id": "x2_task_01", "category": "task_planning", "prompt": "how to install tensorflow on windows step by step"},
    {"id": "x2_task_02", "category": "task_planning", "prompt": "make a 30 day plan to learn dsa"},
    {"id": "x2_task_03", "category": "task_planning", "prompt": "steps to create a github repository and push code"},
    {"id": "x2_task_04", "category": "task_planning", "prompt": "plan a trip to goa for 4 days on a budget"},
    {"id": "x2_task_05", "category": "task_planning", "prompt": "give me a roadmap to become a data scientist"},
    {"id": "x2_task_06", "category": "task_planning", "prompt": "how do i prepare for campus placements, make a timeline"},
    {"id": "x2_task_07", "category": "task_planning", "prompt": "setup guide for docker on ubuntu"},
    {"id": "x2_task_08", "category": "task_planning", "prompt": "create a to do list before submitting my project report"},
    # conversation
    {"id": "x2_conv_01", "category": "conversation", "prompt": "hello there"},
    {"id": "x2_conv_02", "category": "conversation", "prompt": "i'm feeling lonely today"},
    {"id": "x2_conv_03", "category": "conversation", "prompt": "tell me a joke"},
    {"id": "x2_conv_04", "category": "conversation", "prompt": "what's your favorite subject"},
    {"id": "x2_conv_05", "category": "conversation", "prompt": "i failed my test and feel awful"},
    {"id": "x2_conv_06", "category": "conversation", "prompt": "good night, talk tomorrow"},
    {"id": "x2_conv_07", "category": "conversation", "prompt": "can you be my study buddy"},
    {"id": "x2_conv_08", "category": "conversation", "prompt": "i'm bored, what should i do this weekend"},
    # other (quick factual lookups / conversions)
    {"id": "x2_other_01", "category": "other", "prompt": "how many states are there in india"},
    {"id": "x2_other_02", "category": "other", "prompt": "how many sides does a hexagon have"},
    {"id": "x2_other_03", "category": "other", "prompt": "who is the prime minister of japan"},
    {"id": "x2_other_04", "category": "other", "prompt": "convert 10 pounds to kilograms"},
    {"id": "x2_other_05", "category": "other", "prompt": "what is the atomic number of carbon"},
    {"id": "x2_other_06", "category": "other", "prompt": "how many bones are in the human body"},
    {"id": "x2_other_07", "category": "other", "prompt": "which planet is closest to the sun"},
    {"id": "x2_other_08", "category": "other", "prompt": "what is the national animal of india"},
]