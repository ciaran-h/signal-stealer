import os
from tttfast import *
import neuralnets
import numpy as np
import act
import utils
import json

# Clear the screen
os.system("cls")

# Load the outcomes file
# i.e. a dict mapping boards to precomuted best moves
outcomes = None
file_name = "C:\\Users\\Ciaran Hogan\\Desktop\\ttt_board_to_best_moves.json"
with open(file_name, 'r') as outcomes_file:
    outcomes = json.load(outcomes_file)

# Generate input and output samples
print("Generating samples...")
inputs_size = 10**3
#boards = [random_board() for i in range(inputs_size)]
boards = set({})
while len(boards) < inputs_size:
    boards.add(random_board())
inputs = np.array([encode_board(board) for board in boards])
target = np.array([utils.binary_encoding(outcomes[str(board)]['best_moves'],9) for board in boards])

# We no longer need all of the outcomes
del outcomes

# Create and train the neural network
print("Training neural network...")
nn = neuralnets.SimpleFFNNBuilder(learningRate=0.0000000007, seed=987342) \
        .addLayers(len(inputs[0]), actFun=act.LeakyReluAF()) \
        .addLayers(16, 16, 16, actFun=act.LeakyReluAF()) \
        .addLayers(len(target[0]), actFun=act.AtanAF()) \
        .build()
nn.setTrainingData(inputs, target)
nn.train(2*(10**4), graph=True, showOutput=False, showWeights=True)


# Play against the trained neural network
ai_first = True
print("Starting a new tic-tac-toe game...")
while True:
    sep, pad, msg = '-', 7, ' NEW GAME '
    print(sep*pad*2);
    print(sep*(pad-len(msg)//2)+msg+'-'*(pad-len(msg)//2));
    print(sep*pad*2)
    board = empty_board
    while not is_gameover(board):
        if ai_first and board == empty_board:
            data = np.array([encode_board(board)])
            nn.setTrainingData(data, None)
            result = nn.forwardPropagation(data)[0]
            result_valid = [result[i] if is_valid_move(board,i) else float('-inf') for i in range(len(result)-1)]
            ai_move = None
            if max(result_valid) == float('-inf'):
                print('No good results. Making a random move.')
                ai_move = random.choice(get_possible_moves(board))
            else:
                print('Confidence: ', max(result_valid))
                ai_move = result_valid.index(max(result_valid))
                board = make_move(board, ai_move)
            print_board(board)
            if is_gameover(board):
                break
        move = int(input("Move: "))
        board = make_move(board, move)
        print_board(board)
    print('Player ', get_winner(board), ' won!')
    ai_first = not ai_first
