import json
import nltk
import re
from statistics import mean, stdev

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# add_problems = []  # TODO:
# add_problems.extend ([cleanup (problem) for problem in add_problems])
# addition_data = check_freq (add_problems)  # A dict of words and frequencies
#
# sub_problems = []  # TODO:
# sub_problems.extend ([cleanup (problem) for problem in sub_problems])
# subtraction_data = check_freq (sub_problems)  # Dict, as above
# # Serves as 'comparison' value in most functions

"""addition_words = [word for word in addition_data.keys if
                  addition_data[word] > x and not word in subtraction_data.keys ()]"""

# It mostly has support for ngrams (except for the flatten methods sometimes)

problem_data = {"+": [], "-": [], "*": [], "/": []}
"The problem data is a dictionary; an operation name/symbol on one side and massive lists of strings as values"

def iter_flatten(iterable):  # WARNING: Does not work on dicts
    it = iter(iterable)
    for e in it:
        if isinstance(e, (list, tuple)):
            for f in iter_flatten(e):
                yield f
        else:
            yield e


def words_to_list(sentence):
    return list(iter_flatten(sentence.split()))  # Words to list


def dict_combine(hashtable):  # Works on dicts
    """
    TODO: Test
    :param hashtable: A nested dict or associative table of some kind to be unflattened
    :return: An un-nested list of keys and values(as a dictionary)
    """
    flattened = {}
    for nest in hashtable:
        flattened.update(nest)
    return flattened


def writetofile(text, file):
    with open(file, "a") as f:
        f.write(text)
    # Stuff to be written: Testing dataset, scores, later other datasets.


def writetojson(dictobj, file):
    with open(file, mode="r+") as memory:
        s = memory.read().replace("]", "")
        memory.seek(0)
        memory.truncate()
        memory.write(s)
        # Leaves an unclosed parentheses
    with open(file, mode="a") as memory:
        memory.write(", \n")
        json.dump(dictobj, memory, indent=3)
        memory.write("]")
        # Closes that parentheses


def readfromfile(path):
    with open(path, "r") as a_file:
        return json.load(a_file)


def cleanup(data):  # An individual word problem
    """
    TODO: Test
    :param data: A string
    :return: Cleaned up string: Whitespace trimmed, lemmatized, etc.
    """
    lowered = data.lower()
    regexed = re.sub(r"[^\w]", " ", lowered)
    # Could be replaced with something supporting ngrams
    tokenised = nltk.tokenize.word_tokenize(regexed)
    cleaned = [
        token for token in tokenised if token not in set(stopwords.words("english"))
    ]
    lemmatized = [WordNetLemmatizer().lemmatize(word) for word in cleaned]
    return lemmatized


def check_freq(wordlist):  #
    """

    :param wordlist: A nested list of problems, each completely cleaned
    :return: A frequency distribution for the words in that list
    """
    monolith = iter_flatten(wordlist)  # Simple list of words (total)
    return nltk.FreqDist(monolith)
    # pprint.pprint(freq)


def find_unique_words(mainlist, comparisons, x, y):
    """ Very crude; create a better function to use a weighting system based on frequency"""
    # Comparisons is a list of dictionaries, each of which represents a frequency distribution of something to avoid matching
    cleaned = [cleanup(elem) for elem in mainlist]  # A nested list of words
    flattened = iter_flatten(cleaned)
    maindata = check_freq(flattened)
    comparison_keys = []
    for hashmap in comparisons:
        temp_comparison_keys = [k for k in hashmap.keys() if hashmap[k] > y]
        comparison_keys.extend(temp_comparison_keys)
    # All common words in comparison data
    comparison_keys = iter_flatten(comparison_keys)
    uniques = [
        word
        for word in maindata.keys()
        if maindata[word] > x and word not in comparison_keys
    ]  # Unique words to the mainlist of data
    return uniques


def weighted_analysis(mainlist, comparisons, maxscore):
    """

    :param mainlist  The list of test data for the type of problem you want to match
    :param comparisons: The alternatives which you have to avoid scoring
    :param maxscore: The default value to add if the word does not occur in the comparison data
    :return: A dictionary of words with a 'score' based on how often they appear in the mainlist and how rarely they appear in the comparison data
    """
    # Maxscore is the score to give if it does not occur in the comparison data at all
    scores_dict = {}
    cleaned = [cleanup(elem) for elem in mainlist]
    maindata = check_freq(iter_flatten(cleaned))

    # TODO: In case of repeats it should sum frequencies
    clean_comparisons = dict_combine(comparisons)
    # acceptable_candidates = [(word, freq) for word, freq in maindata.items ()]

    for word, mainfreq in dict(maindata).items():
        try:
            comparefreq = clean_comparisons[word]
            score = mainfreq - comparefreq
        # A high score means it occurs very often in the main but not the comparison, and vice versa
        except KeyError:  # If it does not appear in the comparison data
            score = mainfreq + maxscore
        scores_dict.update(word=score)
    return scores_dict
    # Scores_dict is a dictionary of words with their score


def analyse_problem(scoredict, problem, classification, magic_num, deviation):
    """

    :param scoredict: A dictionary of words tied to their frequency, as returned by weighted_analysis()
    :param problem: The string you want to test and classify
    :param classification: The classification you want to test it for, such as if it is an addition problem
    :param magic_num: The conversion factor to go from the score (of a problem) to the probability of it being in that classification
    :param deviation: Taken from the function calculate_magic_num below, it takes the std.
    deviation of the sample dataset
    :return: a certain classification/type (as a string, representing the character of the operation (+,-.*,/)
    """
    # MAGIC NUM is a conversion factor
    cleaned = cleanup(problem)
    problem_score = 0
    for word, freq in scoredict.items():
        if word in cleaned:
            problem_score += freq

    if problem_score in range(magic_num - deviation, magic_num + deviation):
        # print (problem + ' is probably ' + classification)
        return classification
    else:
        # print ('Guido Only Knows')
        return False


def analyse_problem_draft(scoredict, problem, classification):
    # Used for calculating the magic num for a dataset
    cleaned = cleanup(problem)
    problem_score = 0
    for word, freq in scoredict.items():
        if word in cleaned:
            problem_score += freq
    return classification, problem_score


def calculatemagicnum():
    """

    :return: The average score of a successful/confirmed problem, and the standard deviation of that set
    """
    # The magic number is related to the average score
    myscores = readfromfile("scores")
    x = []
    for math_operation, problems in problem_data.items():
        x.extend(
            [
                analyse_problem_draft(
                    myscores[math_operation], problem, math_operation
                )[1]
                for problem in problems
            ]
        )
    return mean(x), stdev(x)


"""
process:
Create scoredict for all 4 operations using massive dataset (done)
Write to files, presumably as JSONS or nested JSONS (done)
analyse the problem using a set of data from each operation (done)
"""


def create_scoredicts():
    # Below describes a sample of the data structure returned
    """
    [  {"+": {"word": "score",
            "word2": "score2"}},
        {"-": {"word": "score",
            "word2": "score2"}},
        {"*": {"word": "score",
            "word2": "score2"}},
        {"/;": {"word": "score",
            "word2": "score2"}}
    ]
    """
    # Creation of the scoredict for all values
    result = {}
    for operation in problem_data.keys():
        # Performs a weighted analysis for each type of problem/operation
        data_copy = problem_data.copy()  # Local var
        # Remove addition, and use as mainlist, use the rest for comparisons
        scoredict = weighted_analysis(data_copy.pop(operation, False)[1], data_copy)
        writetojson(dict(operation=scoredict), "scoredicts.json")
        result.update(operation=scoredict)
    return result


# This includes the order, such as which is the dividend and which is the divisor
def identify_numbers(problem):
    # For now, it identifies them based on bigger or smaller and only supports 2-number problems
    # return an ordered list or tuple, or perhaps a dict
    # Searches for any continuous sequence of integers
    pattern = r"[0-9]*\.[0-9]*"
    # TODO: Test for backslash weirdness
    nums = re.search(pattern, problem).groups()
    # re.search should return a tuple of ints with length=2
    list(nums).sort()
    return list(nums)
    # nums[1]>nums[0]
    # TODO: refactor to use tokenisation for better reliability
    # TODO: Test
    # TODO: Add support for multiple numbers, or not


def isplural(noun):
    """

    :param word: A single noun, string
    :return: Whether it is a plural form or not
    """
    lemma = WordNetLemmatizer().lemmatize(noun, "n")
    return True if noun is not lemma else False


def identify_noun(problem):
    """

    :param problem: A string of words, i.e. a word problem as text
    :return: a single word as a string
    """

    nouns = [
        token
        for token, pos in nltk.pos_tag(nltk.tokenize.word_tokenize(problem))
        if pos.startswith("N")
    ]
    plurals = [word for word in nouns if isplural(word)]
    freq = dict(nltk.FreqDist(plurals))
    # Key with highest value(frequency)
    return max(freq, key=lambda key: freq[key])


def answer(problem, classification):
    num = identify_numbers(problem)
    return exec(num[1] + classification + num[0])
    # return a float or int


def final_answer(problem, classification):
    return str(answer(problem, classification)) + " " + identify_noun(problem)


def integrated_solver(question, scores=readfromfile("scores.json")):
    """

    :param question: A word problem as a string
    :param scores: A scoredict, which can be generated from a file
    :return: A final answer (string)in the form of 'n things'
    """

    # IDEA: Make it show working as well

    classification = False
    for operation in scores.keys():
        # operation_score is loaded from file
        x = analyse_problem(scores[operation], question, operation)
        if x and not classification:
            classification = x
        else:
            raise Exception(
                "We have two possible classifications, and we have no idea which is right. Sorry."
            )

    return final_answer(question, classification)


if __name__ == "__main__":  # Classifies a problem
    print(integrated_solver(input("Question")))

# TODO: Write an HTTPS interface to https://www.mathplayground.com/wpdatabase/Addition_Subtraction_Facts_1.htm and other data sources for additional automated testing


