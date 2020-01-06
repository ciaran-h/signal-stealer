minNumOfActions = 3
maxNumOfActions = 6

bodyParts = ['Head','Nose','LeftEar','RightEar','LeftShoulder','RightShoulder','LeftArm','RightArm']
bodyCodes = ['H','LE','RE','LS','RS','LA','RA']

stealSequence = ['H', 'LS'] #['LE', 'RE']
# If this sequence appears after a steal, then it should be ignored
fakeOutSequence = ['H', 'RE']