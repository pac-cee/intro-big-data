"""Problem 1: Word Frequency Counter
Write a function that takes a string and returns a dictionary with words as keys and their frequency as values.

Problem 2: List Operations
Given two lists, write a function that returns a new list containing only the common elements (without duplicates).

Problem 3: Text Analysis
Write a function that analyzes a text file and returns:

Number of words
Number of unique words
Most common word
Average word length"""

def word_frequency_counter(text):
    words = text.split()
    frequency = {}
    for word in words:
        frequency[word] = frequency.get(word, 0) + 1
    return frequency

def common_elements(list1, list2):
    return list(set(list1) & set(list2))

def text_analysis(file_path):
    with open(file_path, 'r') as file:
        text = file.read()
    words = text.split()
    num_words = len(words)
    num_unique_words = len(set(words))
    most_common_word = max(set(words), key=words.count)
    avg_word_length = sum(len(word) for word in words) / num_words if num_words > 0 else 0
    return {
        "Number of words": num_words,
        "Number of unique words": num_unique_words,
        "Most common word": most_common_word,
        "Average word length": avg_word_length
    }