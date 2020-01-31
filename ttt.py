import random
import copy

def _get_cell_char(cell):

    cellChar = ' '
    if cell == 1:
        cellChar = 'X'
    elif cell == -1:
        cellChar = 'O'

    return cellChar

def random_board():
    '''
    Generates a random, non-gameover board.
    '''
    board = TicTacToe()
    num_of_moves = random.randint(0, 8)
    while num_of_moves > 0:
        pos_moves = board.get_possible_moves()
        non_game_ending_moves = [move for move in pos_moves if not board.is_game_ending_move(move)]
        if len(non_game_ending_moves) == 0:
            return random_board()
        move = random.choice(non_game_ending_moves)
        board.move(move)
        num_of_moves = num_of_moves - 1
    return board

class TicTacToe():

    def __init__(self, board=None, turn=0):
        self._turn = turn
        if board is None:
            self._board = [0 for i in range(9)]
        else:
            self._board = board

    def copy(self):
        return TicTacToe(self._board.copy(), self._turn)

    def _get_score(self, player):
        winner = self.get_winner()
        if player == winner:    return 1
        elif winner == 0:       return 0
        else:                   return -1

    def _negamax(self, depth, player, a=float('-inf'), b=float('inf')):
        '''
        Negamax with alpha-beta pruning.
        '''
        if depth == 0 or self.is_gameover():
            return self._get_score(player)
        
        value = float('-inf')
        for move in self.get_possible_moves():
            new_board = self.branch_move(move)
            move_score = -self._negamax(depth-1, -player, -b, -a)
            value = max(value, move_score)
            a = max(a, value)
            if a >= b:
                break
        return value

    def _minimax(self, depth, player):
        
        if depth == 0 or self.is_gameover():
            return self._get_score(player)
        
        if player == self.get_player():
            value = float('-inf')
            for move in self.get_possible_moves():
                new_board = self.branch_move(move)
                value = max(value, new_board._minimax(depth-1, player))
            return value
        else:
            value = float('inf')
            for move in self.get_possible_moves():
                new_board = self.branch_move(move)
                value = min(value, new_board._minimax(depth-1, player))
            return value

    def get_minimax_moves(self):
        '''
        Returns the best possible move(s) as determined by the minimax algorithm.
        '''
        possible_moves = self.get_possible_moves()
        scores = [self.branch_move(move)._minimax(9, self.get_player()) for move in possible_moves]
        moves = [possible_moves[i] for i in range(len(possible_moves)) if scores[i]==max(scores)]
        return moves
    
    def get_negamax_moves(self):
        '''
        Returns the best possible move(s) as determined by negamax with alpha-beta pruning.
        '''
        possible_moves = self.get_possible_moves()
        scores = [self.branch_move(move)._negamax(9, self.get_player()) for move in possible_moves]
        moves = [possible_moves[i] for i in range(len(possible_moves)) if scores[i]==max(scores)]
        
        return moves
    
    def is_game_ending_move(self, i):
        new_board = self.branch_move(i)
        return new_board.is_gameover()


    def minimax_move(self):
       move = random.choice(self.get_minimax_moves())
       self.move(move)

    def negamax_move(self):
       move = random.choice(self.get_negamax_moves())
       self.move(move)

    def ai_move(self):

        # If the board is empty, go in the top left
        # corner
        if self.is_empty():
            self.move(0)
            return

        best_score = None
        best_move = None
        possible_moves = self.get_possible_moves()
        for move in possible_moves:
            score = self.branch_move(move)._minimax(9, self.get_player())
            if score == 1:
                best_move = move
                break
            if best_score is None or score > best_score:
                best_score = score
                best_move = move
        self.move(move)
        

    def random_move(self):
        move = random.choice(self.get_possible_moves())
        self.move(move)

    def is_gameover(self):
        has_winner = self.get_winner() != 0
        no_free_cells = all([c != 0 for c in self._board])
        return has_winner or no_free_cells

    def is_valid_move(self, i):
        return self._board[i] == 0
    
    def get_possible_moves(self):
        return [i for i in range(9) if self.is_valid_move(i)]

    def get_winner(self):
        
        b = self._board

        # Vertical wins
        if b[0]==b[3]==b[6]!=0:
            return b[0]
        if b[1]==b[4]==b[7]!=0:
            return b[1]
        if b[2]==b[5]==b[8]!=0:
            return b[2]
        
        # Horizontal wins
        if b[0]==b[1]==b[2]!=0:
            return b[0]
        if b[3]==b[4]==b[5]!=0:
            return b[3]
        if b[6]==b[7]==b[8]!=0:
            return b[6]
        
        # Diagonals
        if b[0]==b[4]==b[8]!=0:
            return b[0]
        if b[2]==b[4]==b[6]!=0:
            return b[2]
        
        return 0

    def get_player(self):
        return 1 if self._turn % 2 == 0 else -1

    def move(self, i):

        if not self.is_valid_move(i):
            raise ValueError('Invalid move. Another piece is already at that position.')
        self._board[i] = self.get_player()
        self._turn = self._turn + 1

    def branch_move(self, i):
        if not self.is_valid_move(i):
            raise ValueError('Invalid move. Another piece is already at that position.')
        new_board = self.copy()
        new_board.move(i)
        return new_board

    def is_empty(self):
        return all([c == 0 for c in self._board])

    def draw(self):

        string = '\n '
        for y in range(3):
            for x in range(3):
                i = y * 3 + x
                cell = self._board[i]
                string += ' ' + _get_cell_char(cell) + ' '
                if x < 2:
                    string += '|'
            if y < 2:
                string += '\n --- --- ---\n '
        string += '\n'
        print(string)