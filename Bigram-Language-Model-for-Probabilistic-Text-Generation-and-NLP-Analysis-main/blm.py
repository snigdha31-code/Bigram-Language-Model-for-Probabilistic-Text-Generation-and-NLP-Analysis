import nltk  # Import the Natural Language Toolkit
from nltk.corpus import brown, stopwords  # Import the Brown corpus and stopwords
from nltk.probability import ConditionalFreqDist  # Import the ConditionalFreqDist class

# Ensure necessary NLTK resources are available
nltk.download('brown')
nltk.download('stopwords')# Download the Brown corpus and stopwords
nltk.download('punkt')

# Load stopwords
stop_words = set(stopwords.words('english'))

# Extract words from the Brown corpus, removing stopwords and non-alphabetic tokens
import nltk # Import the Natural Language Toolkit
from nltk.corpus import brown# Import the Brown corpus
from nltk import bigrams# Import the bigrams function


nltk.download('brown')# Download the Brown corpus

def extract_corpus_words():
   # Extract words from the Brown corpus
    corpus_words = [word.lower() for word in brown.words() if word.isalpha()]# Extract words from the Brown corpus
    return corpus_words

def generate_bigram_statistics(corpus_words):
   # Generate bigram and word frequency statistics from the corpus
    bigram_frequency = {}
    word_frequency = {}

    for first_word, second_word in bigrams(corpus_words):
        if first_word not in bigram_frequency:
            bigram_frequency[first_word] = {}
        if second_word not in bigram_frequency[first_word]:
            bigram_frequency[first_word][second_word] = 0
        bigram_frequency[first_word][second_word] += 1

        word_frequency[first_word] = word_frequency.get(first_word, 0) + 1

    return bigram_frequency, word_frequency

def compute_sentence_probability(user_input, bigram_frequency, word_frequency):
  # Compute the probability of a sentence using bigram statistics
    words_in_sentence = [word.lower() for word in user_input.split() if word.isalpha()]# Extract words from the user input
    sentence_bigram_list = list(bigrams(words_in_sentence))

    sentence_probability = 1.0
    INITIAL_PROB = 0.25
    TERMINAL_PROB = 0.25

    print("\nComputed Bigram Probabilities:")
    for previous, current in sentence_bigram_list:
        bigram_count = bigram_frequency.get(previous, {}).get(current, 0)# Get the bigram count
        previous_word_count = word_frequency.get(previous, 0)# Get the previous word count

 
        if previous_word_count == 0:
            bigram_prob = 0
        else:
            bigram_prob = bigram_count / previous_word_count


        if previous in {".", "!", "?"}:
            bigram_prob = INITIAL_PROB
        if current in {".", "!", "?"}:
            bigram_prob = TERMINAL_PROB

        print(f"Bigram: ({previous}, {current}), Probability: {bigram_prob}")# Display the bigram and its probability
        sentence_probability *= bigram_prob

    return sentence_probability

def execute_bigram_analysis():
# Execute the bigram analysis
    user_sentence = input("Enter a sentence: ").lower()# Get user input
    print(f"Processed Sentence: {user_sentence}")#  Display the processed sentence

    corpus_words = extract_corpus_words()# Extract words from the Brown corpus
    bigram_data, word_data = generate_bigram_statistics(corpus_words)#  Generate bigram and word frequency statistics from the corpus

    final_probability = compute_sentence_probability(user_sentence, bigram_data, word_data)# Compute the probability of a sentence using bigram statistics
    print(f"\nSentence: {user_sentence}")# Display the sentence
    print(f"Computed Probability: {final_probability}")# Display the computed probability

if __name__ == "__main__":
    execute_bigram_analysis()
words_list = [word.lower() for word in brown.words() if word.isalpha() and word.lower() not in stop_words]

# Generate bigrams from the filtered words list
paired_words = list(nltk.bigrams(words_list))

# Create a conditional frequency distribution for bigrams
bigram_distribution = ConditionalFreqDist(paired_words)

def predict_next_word(current_term, bigram_distribution):  
    """Predict the next word based on the current word using bigram probabilities."""
    if current_term in bigram_distribution:
        frequent_words = bigram_distribution[current_term].most_common(3)  # Get 3 most common words
        return frequent_words
    return None

def construct_sentence(bigram_distribution):
    """Interactively construct a sentence by predicting the most probable next words."""
    
    initial_term = input("Enter a starting word: ").lower()  # Get the initial word from the user

    if initial_term == 'quit':
        return
    
    phrase = [initial_term]

    while True:
        suggestions = predict_next_word(initial_term, bigram_distribution)  # Get the next word suggestions

        if suggestions:
            print(f"\nCurrent word: {initial_term}")
            print("Choose the next word:")

            for index, (next_term, count) in enumerate(suggestions, 1):  # Display the suggestions
                probability = count / sum(bigram_distribution[initial_term].values())  # Calculate the probability
                print(f"{index}) {next_term} P({initial_term} -> {next_term}) = {probability:.2f}")  # Display probability

            print("4) QUIT")  # Display the quit option

            selection = input("Select an option (1-4): ").strip().lower()  # Get user selection

            if selection == "4" or selection == "quit":
                break
            elif selection in ["1", "2", "3"]:
                chosen_word = suggestions[int(selection) - 1][0]  # Get the chosen word
            else:
                chosen_word = suggestions[0][0]  # Default to the first suggestion if input is invalid

            phrase.append(chosen_word)  # Append the chosen word to the phrase
            initial_term = chosen_word  # Update the initial term

            print("\nUpdated sentence:", " ".join(phrase))  # Display the updated sentence

        else:
            print("\nThe word is not found in the corpus. Choose an action:")  # Display options
            print("a) ASK AGAIN")
            print("b) QUIT")

            response = input().strip().lower()  # Get user response
            if response in ['b', 'quit']:
                break
            elif response == 'a':
                initial_term = input("Enter a different word: ").lower()  # Get a different word
                phrase = [initial_term]  
            else:
                print("Invalid selection. Try again.")

    print("\nFinal constructed sentence:", " ".join(phrase))  # Display the final sentence

if __name__ == "__main__":
    construct_sentence(bigram_distribution)
