from tttfast import *
import ttt
import time, random
import json

'''
print('Test 1')
start = time.time()
board = empty_board
for i in range(1000000):
    make_move(board, random.randint(0, 8))
print('Done in ', time.time() - start, ' seconds')


print('Test 2')
start = time.time()
board = ttt.TicTacToe()
for i in range(1000000):
    board.branch_move(random.randint(0, 8))
print('Done in ', time.time() - start, ' seconds')
'''

'''
outcomes = None
file_name = "C:\\Users\\Ciaran Hogan\\Desktop\\ttt_board_to_best_moves.json"
with open(file_name, 'r') as outcomes_file:
    outcomes = json.load(outcomes_file)

board = empty_board
while not is_gameover(board):
    #board = minimax_move(board)
    all_best_moves = outcomes[str(board)]['best_moves']
    ai_move = random.choice(all_best_moves)
    board = make_move(board, ai_move)
    print_board(board)
    move = int(input("Move: "))
    board = make_move(board, move)
    print_board(board)

print('Player ', get_winner(board), ' won!')
'''

for i in range(10):
    print(print_board(random_board()))