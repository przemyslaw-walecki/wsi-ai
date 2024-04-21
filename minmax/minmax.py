'''
MinMax decision rule 
Implemented for tic-tac-toe
'''

from copy import deepcopy
from matplotlib import pyplot as plt
import numpy as np
class Player:
    def __init__(self, sign:str) -> None:
        self.sign = sign


class State:
    def __init__(self, turn='X', board=None) -> None:
        self.board = board if board is not None else [[' ' for _ in range(3)] for __ in range(3)]
        self.turn = turn
    
    def is_terminal(self):
        if self.check_winner('X') or self.check_winner('O'):
            return True
        return all(all(cell != ' ' for cell in row) for row in self.board)
    
    def display(self):
        for row in self.board:
            print("|".join(row))
            print("-----") 
        print('===========')
    
    def check_winner(self, player_sign):
        for row in self.board:
            if all(cell == player_sign for cell in row):
                return True

        for col in range(3):
            if all(self.board[row][col] == player_sign for row in range(3)):
                return True

        if all(self.board[i][i] == player_sign for i in range(3)) or \
           all(self.board[i][2 - i] == player_sign for i in range(3)):
            return True
        
        return False
    
    
    def get_available_moves(self):
        return [(i, j) for i in range(3) for j in range(3) if self.board[i][j] == ' ']
    
    def make_move(self, move):
        i, j = move
        self.board = deepcopy(self.board)
        self.board[i][j] = self.turn
        self.turn = 'O' if self.turn == 'X' else 'X'


def evaluate(state:State):
    if state.check_winner('X'):
        return 100
    elif state.check_winner('O'):
        return -100
    else:
        return calculate_line_score(state, 'X') - calculate_line_score(state, 'O')


def calculate_line_score(state:State, player_sign):
    weights = [1, 10, 100]
    return sum(weight * count_lines(state, player_sign, length) for weight, length in enumerate(weights, 1))

    
def count_lines(state:State, player_sign, length):
    lines = [state.board[row] for row in range(3)] + \
            [[state.board[row][col] for row in range(3)] for col in range(3)] + \
            [[state.board[i][i] for i in range(3)], [state.board[i][2 - i] for i in range(3)]]

    return sum(count_occurrences(player_sign, length, line) for line in lines)


def count_occurrences(player_symbol, length, lines):
    return sum(1 for line in lines if line.count(player_symbol) == length and line.count(' ') == 3 - length)

def min_max(state: State, depth, alpha, beta, maximizing_player):
    if depth == 0 or state.is_terminal():
        return evaluate(state), None

    best_move = None
    moves = sorted(state.get_available_moves(), key= lambda k: np.random.random())
    if maximizing_player:
        max_eval = float('-inf')
        for move in moves:
            new_state = State(state.turn, state.board)
            new_state.make_move(move)
            eval = min_max(new_state, depth - 1, alpha, beta, False)[0] 
            if eval > max_eval:
                max_eval = eval
                best_move = move
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval, best_move
    else:
        min_eval = float('inf')
        for move in moves:
            new_state = State(state.turn, state.board)
            new_state.make_move(move)
            eval = min_max(new_state, depth - 1, alpha, beta, True)[0]
            if eval < min_eval:
                min_eval = eval
                best_move = move
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval, best_move


class Game:
    def __init__(self, player1:Player, player2:Player) -> None:
        self.state = State()
        self.player1 = player1
        self.player2 = player2

    def play_2minmax(self, p1depth, p2depth):
        while not self.state.is_terminal():
            self.state.display()
            if self.state.turn == 'X':
                move = min_max(self.state, p1depth, float('-inf'), float('inf'), True)[1]
                self.state.make_move(move)
            else:
                move = min_max(self.state, p2depth, float('-inf'), float('inf'), False)[1]
                self.state.make_move(move)
        
        self.state.display()
        print(f'###Game result: {self.who_won()}###')
        print('===========')
    
    def who_won(self):
        if self.state.is_terminal():
            if self.state.check_winner('X'):
                return 'X won'
            if self.state.check_winner('O'):
                return 'O won'
            else:
                return 'Draw'
        return None

    
    def reset(self):
        self.state = State()

                
        
                

def main():
    p1 = Player('X')
    p2 = Player('O')
    game = Game(p1, p2)
    p1depth = 9
    p2depth = 9

    results = {'X won': 0, 'O won': 0, 'Draw': 0}
    for _ in range(10):
        game.play_2minmax(p1depth, p2depth)
        results[game.who_won()] += 1
        game.reset()
    print(f'P1:{p1depth} P2: {p2depth}, {results}')
    
if __name__ == "__main__":
    main()