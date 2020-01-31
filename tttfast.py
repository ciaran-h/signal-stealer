import random

cell_to_char = [' ', 'X', 'O']

player_1_wins = [
    0b_0_000000_000000_010101,
    0b_0_000000_010101_000000,
    0b_0_010101_000000_000000,
    0b_0_000001_000001_000001,
    0b_0_000100_000100_000100,
    0b_0_010000_010000_010000,
    0b_0_010000_000100_000001,
    0b_0_000001_000100_010000
]

player_2_wins = [
    0b_0_000000_000000_101010,
    0b_0_000000_101010_000000,
    0b_0_101010_000000_000000,
    0b_0_000010_000010_000010,
    0b_0_001000_001000_001000,
    0b_0_100000_100000_100000,
    0b_0_100000_001000_000010,
    0b_0_000010_001000_100000
]

turn_mask = 0b_1_0000_0000_0000_0000_00
board_mask = 0b_0_1111_1111_1111_1111_11
empty_board = 0b_0_0000_0000_0000_0000_00
ones_mask = 0b_1_1111_1111_1111_1111_11

def encode_board(board):
    return [player if (player:=get_cell(board,i))!=2 else -1 for i in range(9)]

def get_binary_encoding(board) :
    # Isolate the board and pad zeros to the left so all
    # binary representations are the same size
    # Note: the [3:] removes the '0b' and the turn info
    board_binary = bin(board)[3:].rjust(18, '0')
    return [0 if c=='0' else 1 for c in board_binary]

# TODO: use outcomes file and search for boards with non empty
# lists of moves
def random_board():
    '''
    Generates a random tic-tac-toe board that is NOT
    gameover.
    '''
    board = empty_board
    num_of_turns = random.randint(0, 8)
    while num_of_turns > 0:
        pos_moves = get_possible_moves(board)
        non_game_ending_moves = [move for move in pos_moves if not is_gameover(make_move(board, move))]
        if len(non_game_ending_moves) == 0:
            # This path is a deadend. Try another.
            return random_board()
        move = random.choice(non_game_ending_moves)
        board = make_move(board, move)
        num_of_turns = num_of_turns - 1
    return board

def get_score(board, player):
    winner = get_winner(board)
    if player == winner:    return 1
    elif winner == 0:       return 0
    else:                   return -1

def get_possible_moves(board):
    return [i for i in range(9) if is_valid_move(board, i)]

def get_turn(board):
    return ((board & turn_mask) >> 18) + 1

def make_move(board, index):
    assert 0 <= index <= 8
    # Get the position of the bit to change
    bit_index = index * 2 + get_turn(board)-1
    # Set the bit at that position to 1
    mask = 0b1<<bit_index   # Set the <bit_index>-th index to 1, o/w 0
    board = board | mask
    # not the first bit only
    # i.e. 1xx...x -> 0xx...x and 0xx...x -> 1xx..x
    board = board ^ turn_mask
    return board

def is_valid_move(board, index):
    return get_cell(board, index) == 0

def get_cell(board, index):
    assert 0 <= index <= 8
    # Create mask that is 1 at index*2 and index*2+1
    mask = 0b11 << (index*2)
    # Isolate for the values in the 1s' position
    # 00 - 0, 01 - 1, 10 - 2, 11 - undefined (shouldn't ever happen)
    return (board & mask) >> (index*2)

def get_winner(board):

    # Check if player 1 has won
    for mask in player_1_wins:
        if board & mask == mask:
            return 1
    
    # Check if player 2 has won
    for mask in player_2_wins:
        if board & mask == mask:
            return 2
    
    # Neither have won
    return 0

def has_free_cells(board):
    '''
    Returns true if there are any free cells on the board.
    '''
    # TODO: try and make this work
    '''
    # Isolate for the board information only
    board = board & board_mask
    # Shift the board left and right so we can compare
    # the neighbouring X and O values
    shifted_left = board<<1
    shifted_right = board>>1
    # This (hopefully) works because for each of the 'cells'
    # only one bit can be 1 
    # So as long as the bit to the left or right is 1, then
    # that place on the board is taken
    combined = shifted_left^board | shifted_right^board

    if combined >= 0b_0_100000_000000_000000:
        combined = combined // 2
    
    print(bin(combined))
    # If all of the 'cells' are 1, then the board is full
    return not combined == 0b11111_111111_111111
    '''
    return any([get_cell(board, i)==0 for i in range(9)])

def is_empty(board):
    '''
    Returns true if there are any free cells on the board.
    '''
    return all([get_cell(board, i)==0 for i in range(9)])


def is_gameover(board):
    '''
    Returns true if either player has won, or if there are
    no more free cells.
    '''
    return not has_free_cells(board) or get_winner(board)!=0

def _minimax(board, player):
    
    if is_gameover(board):
        return get_score(board, player)
    
    if player == get_turn(board):
        value = float('-inf')
        for move in get_possible_moves(board):
            new_board = make_move(board, move)
            value = max(value, _minimax(new_board, player))
        return value
    else:
        value = float('inf')
        for move in get_possible_moves(board):
            new_board = make_move(board, move)
            value = min(value, _minimax(new_board, player))
        return value

def get_minimax_moves(board):
    '''
    Returns the best possible move(s) as determined by the minimax algorithm.
    '''
    
    if is_empty(board):
        return [0, 1, 2, 3, 4, 5, 6, 7, 8]
    
    possible_moves = get_possible_moves(board)
    scores = [_minimax(make_move(board, move), get_turn(board)) for move in possible_moves]
    moves = [possible_moves[i] for i in range(len(possible_moves)) if scores[i]==max(scores)]
    return moves

def minimax_move(board):
    move = random.choice(get_minimax_moves(board))
    return make_move(board, move)

def print_board(board):
    '''
    Prints a nicely formatted board to the terminal.
    '''
    string = '\n '
    for y in range(3):
        for x in range(3):
            # Invert the y so it is draw from bottom to top
            i = (2-y) * 3 + x
            cell = get_cell(board, i)
            string += ' ' + cell_to_char[cell] + ' '
            if x < 2:
                string += '|'
        if y < 2:
            string += '\n --- --- ---\n '
    string += '\n'
    print(string)