from tttfast import *
import json
import os

def _compute_outcomes_helper(board):
    '''
    A helper function for compute_possibilities(). Computes all possible boards and the
    best move(s) to make in that instance.
    '''

    if is_gameover(board):
        result = {
            "is_game_over" : True,
            "result" : get_winner(board),
        }
        return result
    else:
        outcomes = {
            "is_game_over" : False,
            "best_moves" : get_minimax_moves(board), 
            "outcomes" : []
        }
        for move in range(9):
            if is_valid_move(board, move):
                new_board = make_move(board, move)
                outcomes['outcomes'].append(_compute_outcomes_helper(new_board))
            else:
                outcomes['outcomes'].append(None)
        return outcomes

def compute_outcomes():
    '''
    Computes all possible boards and the best move(s) to make in
    that instance.
    '''
    return _compute_outcomes_helper(empty_board)

def _compute_boards_helper(board_to_best_moves, board):
    '''
    A helper function for compute_boards().
    '''
    if not is_gameover(board):
        board_to_best_moves[board] = {"best_moves" : get_minimax_moves(board)}
        for move in range(9):
            if is_valid_move(board, move):
                new_board = make_move(board, move)
                _compute_boards_helper(board_to_best_moves, new_board)

def compute_boards():
    '''
    Returns a dictionary mapping boards to the best move(s) the
    current player should make.
    '''
    board_to_best_moves = {}
    _compute_boards_helper(board_to_best_moves, empty_board)
    return board_to_best_moves


if __name__ == "__main__":
    
    print('Where would you like to save the outcomes json?')
    file_name = None
    is_valid_path = False
    while not is_valid_path:
        file_name = input('File Path: ')
        is_valid_path = file_name is not None \
            and os.path.exists(os.path.dirname(file_name)) \
            and os.path.splitext(file_name)[1] == '.json'

    print('Computing outcomes...', end='')
    outcomes = compute_boards()
    #outcomes = compute_outcomes()
    print(' Done!')

    print('Saving to ' + file_name)
    with open(file_name, 'w') as outcomes_file:
        json.dump(outcomes, outcomes_file)

    print('Done. Exiting...')