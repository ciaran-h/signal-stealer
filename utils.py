
# https://stackoverflow.com/questions/36016174/how-to-check-if-a-list-is-in-another-list-with-the-same-order-python
def containsSubsequence(sequence, subsequence):
	'''
	Returns true if the subsequence is contained within
    the sequence.
	'''
	for i in range(len(sequence) - len(subsequence) + 1):
		if subsequence == sequence[i:i+len(subsequence)]:
			return True
	return False

def one_hot(index, size):
	'''
	Returns an array that is 1 at the specified index and
	zero everywhere else.
	'''
	return [1 if i==index else 0 for i in range(size)]

def binary_encoding(indices, size):
	'''
	Returns an array that is 1 at the specified indices and
	zero everywhere else.
	'''
	return [1 if i in indices else 0 for i in range(size)]