import const
import utils
import random
import numpy as np

# Input: minLength = 2, maxLength = 3
# Output: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
def randomBodySequence(minLength, maxLength):
    '''
    Generates a random sequence of actions where if the (index % len(bodyCodes))-th
    action is 1, then the corresponding body code is the action taken.
    '''

    assert minLength >= 2
    assert maxLength >= minLength

    # Determine number of actions
    length = random.randint(minLength, maxLength)
    # Create array of zeros
    sequence = [0 for i in range(maxLength * len(const.bodyCodes))]

    # Set actions to 1
    for i in range(length):
        action = random.randint(0, len(const.bodyCodes)-1)
        sequence[i * len(bodyCodes) + action] = 1

    return sequence

# Input: ['H', 'LS', 'RA']
# Output: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
def bodyCodeSequenceToIntCode(actions):
    '''
    Converts a sequence of actions into a list where if the (index % len(bodyCodes))-th
    element is 1, then the corresponding body code was the action taken.
    '''

    assert len(actions) in range(const.minNumOfActions, const.maxNumOfActions + 1)

    # Create array of zeros
    sequence = [0 for i in range(const.maxNumOfActions * len(const.bodyCodes))]

    # Set actions to 1
    for i in range(len(actions)):
        action = const.bodyCodes.index(actions[i])
        sequence[i * len(const.bodyCodes) + action] = 1

    return sequence

def randomIndex(a, b):
    if a >= b:
        return a
    else:
        return random.randint(a, b)

def isStealAndNotFakeOut(seq):

    endIndex = None
    for i in range(len(seq) - len(const.stealSequence) + 1):
        if const.stealSequence == seq[i:i+len(const.stealSequence)]:
            endIndex = i + len(const.stealSequence)
            break
    
    if endIndex is None:
        return False

    for i in range(endIndex, len(seq)):
        if const.fakeOutSequence == seq[i:i+len(const.fakeOutSequence)]:
            return False
    return True

def isStealAndFakeOut(seq):

    endIndex = None
    for i in range(len(seq) - len(const.stealSequence) + 1):
        if const.stealSequence == seq[i:i+len(const.stealSequence)]:
            endIndex = i + len(const.stealSequence)
            break
    
    if endIndex is None:
        return False

    for i in range(endIndex, len(seq)):
        if const.fakeOutSequence == seq[i:i+len(const.fakeOutSequence)]:
            return True
    return False

def isNotSteal(seq):
    return not utils.containsSubsequence(seq, const.stealSequence)

def randomStealWithFakeOut():

    sequence = []
    length = random.randint(const.minNumOfActions, const.maxNumOfActions)
    # Ensure we have enough room for both sequences
    length = max(length, len(const.stealSequence) + len(const.fakeOutSequence))
    
    # Insert random actions ensuring that they do not contain
    # the steal sequence
    
    index = randomIndex(0, length - 1 - len(const.stealSequence) - len(const.fakeOutSequence))
    count = index - 0
    while len(sequence) < count or utils.containsSubsequence(sequence, const.stealSequence):
        sequence = random.choices(const.bodyCodes, k = count)
    
    # Insert steal sequence
    sequence.extend(const.stealSequence)

    # Insert random actions ensuring that they do not contain the
    # fakeout sequence
    index = randomIndex(len(sequence), length - 1 - len(const.fakeOutSequence))
    count = index - len(sequence)
    subsequence = []
    while len(subsequence) < count or utils.containsSubsequence(subsequence, const.fakeOutSequence):
        subsequence = random.choices(const.bodyCodes, k = count)
    sequence.extend(subsequence)

    # Insert the fakeout sequence
    sequence.extend(const.fakeOutSequence)

    # Insert random actions ensuring that they do not contain the
    # fakeout sequence
    subsequence = []
    index = length
    count = index - len(sequence)
    while len(subsequence) < count:
        subsequence = random.choices(const.bodyCodes, k = count)
    sequence.extend(subsequence)

    return sequence

def randomStealWithoutFakeOut():

    sequence = []
    length = random.randint(const.minNumOfActions, const.maxNumOfActions)

    # Ensure we have enough room for both sequences
    length = max(length, len(const.stealSequence))

    # Insert random actions ensuring that they do not contain
    # the steal sequence
    index = randomIndex(0, length - 1 - len(const.stealSequence))
    count = index - 0
    while len(sequence) < index or utils.containsSubsequence(sequence, const.stealSequence):
        sequence = random.choices(const.bodyCodes, k = count)

    # Insert steal sequence
    sequence.extend(const.stealSequence)

    # Insert random actions ensuring that they do not contain the
    # fakeout sequence
    subsequence = []
    index = length
    count = index - len(sequence)
    while len(subsequence) < count or utils.containsSubsequence(subsequence, const.fakeOutSequence):
        subsequence = random.choices(const.bodyCodes, k = count)
    sequence.extend(subsequence)

    return sequence

def randomNoSteal():

    sequence = []
    length = random.randint(const.minNumOfActions, const.maxNumOfActions)

    while len(sequence) < length or utils.containsSubsequence(sequence, const.stealSequence):
        sequence = random.choices(const.bodyCodes, k = length)

    return sequence

def generateSampleData(samples, probOfStealWithFakeOut, probOfStealWithoutFakeOut):

    assert samples > 0

    data = []

    for i in range(int(samples)):

        sequence = []
        length = random.randint(const.minNumOfActions, const.maxNumOfActions)
        if random.uniform(0, 1) < probOfStealWithFakeOut:
            data.append((randomStealWithFakeOut(), 0.0))
        elif random.uniform(0, 1) < probOfStealWithoutFakeOut:
            data.append((randomStealWithoutFakeOut(), 1.0))
        else:
            data.append((randomNoSteal(), 0.0))

    inputs = np.array([bodyCodeSequenceToIntCode(sequence) for sequence, expected in data])
    output = np.array([[expected for sequence, expected in data]]).T
    
    return inputs, output
