
# https://stackoverflow.com/questions/36016174/how-to-check-if-a-list-is-in-another-list-with-the-same-order-python
def containsSubsequence(sequence, subsequence):
  for i in range(len(sequence) - len(subsequence) + 1):
    if subsequence == sequence[i:i+len(subsequence)]:
        return True
  return False