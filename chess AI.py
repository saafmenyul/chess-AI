import chess
import chess.engine
import random
import math
import numpy as np
from collections import defaultdict
import pickle
import os

class ChessMCTSNode:
    def __init__(self, board, parent=None, move=None):
        self.board = board.copy()
        self.parent = parent
        self.move = move
        self.children = []
        self.wins = 0
        self.visits = 0
        self.untried_moves = list(self.board.legal_moves)
        
    def is_fully_expanded(self):
        return len(self.untried_moves) == 0
    
    def is_terminal_node(self):
        return self.board.is_game_over()
    
    def best_child(self, exploration_weight=1.4):
        """Выбор лучшего дочернего узла по UCB1"""
        choices_weights = [
            (child.wins / child.visits) + 
            exploration_weight * math.sqrt(2 * math.log(self.visits) / child.visits)
            for child in self.children
        ]
        return self.children[np.argmax(choices_weights)]
    
    def expand(self):
        """Расширение дерева - добавление нового хода"""
        move = self.untried_moves.pop()
        new_board = self.board.copy()
        new_board.push(move)
        child_node = ChessMCTSNode(new_board, self, move)
        self.children.append(child_node)
        return child_node
    
    def update(self, result):
        """Обновление статистики узла"""
        self.visits += 1
        self.wins += result

class SelfLearningChessBot:
    def __init__(self, time_limit=2.0, exploration_weight=1.4):
        self.time_limit = time_limit
        self.exploration_weight = exploration_weight
        self.q_values = defaultdict(lambda: defaultdict(float))
        self.visit_counts = defaultdict(lambda: defaultdict(int))
        
    def get_state_key(self, board):
        """Преобразование доски в ключ для Q-таблицы"""
        return board.fen()
    
    def rollout_policy(self, board):
        """Случайная политика для симуляции"""
        return random.choice(list(board.legal_moves))
    
    def rollout(self, board):
        """Проведение случайной симуляции до конца игры"""
        current_board = board.copy()
        while not current_board.is_game_over():
            move = self.rollout_policy(current_board)
            current_board.push(move)
        
        # Оценка результата
        if current_board.is_checkmate():
            return 1.0 if current_board.turn != board.turn else 0.0
        else:  # Ничья
            return 0.5
    
    def mcts(self, root_board, iterations=1000):
        """Алгоритм Монте-Карло для поиска по дереву"""
        root_node = ChessMCTSNode(root_board)
        
        for _ in range(iterations):
            node = root_node
            board = root_board.copy()
            
            # 1. Selection - выбор узла для расширения
            while not node.is_terminal_node() and node.is_fully_expanded():
                node = node.best_child(self.exploration_weight)
                board.push(node.move)
            
            # 2. Expansion - расширение дерева
            if not node.is_terminal_node() and not node.is_fully_expanded():
                node = node.expand()
                board.push(node.move)
            
            # 3. Simulation - симуляция игры
            result = self.rollout(board)
            
            # 4. Backpropagation - обновление статистики
            while node is not None:
                # Преобразование результата к перспективе текущего игрока
                if node.parent is None:
                    adjusted_result = result
                else:
                    adjusted_result = 1 - result if board.turn != node.board.turn else result
                
                node.update(adjusted_result)
                
                # Обновление Q-значений
                state_key = self.get_state_key(node.board)
                if node.move:
                    move_key = node.move.uci()
                    current_q = self.q_values[state_key][move_key]
                    current_visits = self.visit_counts[state_key][move_key]
                    
                    # Обновление среднего значения
                    new_q = (current_q * current_visits + adjusted_result) / (current_visits + 1)
                    self.q_values[state_key][move_key] = new_q
                    self.visit_counts[state_key][move_key] = current_visits + 1
                
                node = node.parent
                if node and node.move:
                    board.pop()
        
        # Выбор лучшего хода на основе статистики
        if root_node.children:
            best_child = max(root_node.children, key=lambda child: child.visits)
            return best_child.move
        else:
            return random.choice(list(root_board.legal_moves))
    
    def get_move(self, board):
        """Получение хода от бота"""
        return self.mcts(board)
    
    def self_play_game(self):
        """Сыграть одну игру против самого себя"""
        board = chess.Board()
        moves = []
        
        while not board.is_game_over() and len(moves) < 200:  # Ограничение на длину игры
            move = self.get_move(board)
            moves.append((board.fen(), move.uci()))
            board.push(move)
        
        # Определение результата
        if board.is_checkmate():
            result = 1.0 if board.turn == chess.BLACK else 0.0  # Победа черных/белых
        else:  # Ничья
            result = 0.5
        
        return moves, result
    
    def train(self, num_games=100):
        """Обучение бота через самосыгрывание"""
        print(f"Начало обучения на {num_games} игр...")
        
        for game_num in range(num_games):
            moves, result = self.self_play_game()
            
            # Обновление Q-значений на основе результата игры
            for state_fen, move_uci in moves:
                current_q = self.q_values[state_fen][move_uci]
                current_visits = self.visit_counts[state_fen][move_uci]
                
                # Обновление с учетом результата игры
                new_q = (current_q * current_visits + result) / (current_visits + 1)
                self.q_values[state_fen][move_uci] = new_q
                self.visit_counts[state_fen][move_uci] = current_visits + 1
            
            if (game_num + 1) % 10 == 0:
                print(f"Сыграно игр: {game_num + 1}")
    
    def save_model(self, filename="chess_bot_model.pkl"):
        """Сохранение обученной модели"""
        model_data = {
            'q_values': dict(self.q_values),
            'visit_counts': dict(self.visit_counts)
        }
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Модель сохранена в {filename}")
    
    def load_model(self, filename="chess_bot_model.pkl"):
        """Загрузка обученной модели"""
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                model_data = pickle.load(f)
            self.q_values = defaultdict(lambda: defaultdict(float), model_data['q_values'])
            self.visit_counts = defaultdict(lambda: defaultdict(int), model_data['visit_counts'])
            print(f"Модель загружена из {filename}")
        else:
            print(f"Файл {filename} не найден")

def play_against_bot(bot, board=None):
    """Игра против обученного бота"""
    if board is None:
        board = chess.Board()
    
    print("Шахматный бот готов к игре!")
    print("Введите ход в формате UCI (например, 'e2e4') или 'quit' для выхода")
    
    while not board.is_game_over():
        print("\n" + str(board))
        print(f"Текущий ход: {'Белые' if board.turn else 'Черные'}")
        
        if board.turn == chess.WHITE:
            # Ход человека
            while True:
                user_input = input("Ваш ход: ").strip()
                if user_input.lower() == 'quit':
                    return
                
                try:
                    move = chess.Move.from_uci(user_input)
                    if move in board.legal_moves:
                        board.push(move)
                        break
                    else:
                        print("Недопустимый ход! Попробуйте снова.")
                except ValueError:
                    print("Неверный формат! Используйте UCI нотацию (например, 'e2e4')")
        else:
            # Ход бота
            print("Бот думает...")
            bot_move = bot.get_move(board)
            board.push(bot_move)
            print(f"Бот сделал ход: {bot_move.uci()}")
    
    # Конец игры
    print("\n" + str(board))
    if board.is_checkmate():
        winner = "Белые" if board.turn == chess.BLACK else "Черные"
        print(f"Шах и мат! Победили {winner}!")
    else:
        print("Ничья!")

# Пример использования
if __name__ == "__main__":
    # Создание и обучение бота
    bot = SelfLearningChessBot()
    
    # Попытка загрузить существующую модель
    bot.load_model()
    
    # Обучение (можно пропустить, если модель уже обучена)
    train_new = input("Обучить новую модель? (y/n): ").lower().strip()
    if train_new == 'y':
        num_games = int(input("Количество игр для обучения: "))
        bot.train(num_games)
        bot.save_model()
    
    # Игра против бота
    play_against_bot(bot)