import numpy as np
import tensorflow as tf
import pygame
import random
import string
from tkinter import Tk, simpledialog
from nltk.corpus import words as nltk_words
from rapidfuzz import fuzz
import nltk
import spacy
from spacy.lang.en import English
from pygame.locals import QUIT
from threading import Thread, Event, Lock
from transformers import BertTokenizer, TFBertForSequenceClassification, GPT2Tokenizer, TFGPT2Model

# Load BERT model and tokenizer
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')
bert_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Load GPT-2 model and tokenizer
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt2_model = TFGPT2Model.from_pretrained('gpt2')

# Colors
WHITE = (255, 255,255)
BLACK = (0,0,0)
RED = (255,0,0)
GREEN = (0,255,0,128)  # Green with 50% transparency

AI_UPDATE_EVENT = pygame.USEREVENT + 1

# Download NLTK resources
try:
    nltk_words.words()
except LookupError:
    print("Downloading NLTK words dataset...")
    nltk.download('words')

# Download spaCy model
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    print("Downloading spaCy model...")
    spacy.cli.download('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

# TrieNode class for trie data structure
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False

# Trie class for word storage and search
class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True

    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end_of_word

# Function to generate a random grid of letters
def generate_grid(grid_size, english_letter_ratio):
    english_letters = string.ascii_uppercase
    total_cells = grid_size * grid_size

    # Generate a grid with random letters
    grid = [
        [random.choice(english_letters) if random.random() < english_letter_ratio else random.choice(string.ascii_uppercase)
         for _ in range(grid_size)] for _ in range(grid_size)
    ]

    return grid

# Function to draw the grid on the screen
def draw_grid(screen, grid, cell_size, selected_cells):
    for row, row_content in enumerate(grid):
        for col, cell_content in enumerate(row_content):
            x, y, width, height = col * cell_size, row * cell_size, cell_size, cell_size
            pygame.draw.rect(screen, WHITE, (x, y, width, height), 2)
            font_size = min(36, cell_size -8)
            font = pygame.font.Font(None, font_size)
            text = font.render(cell_content, True, BLACK)
            screen.blit(text, (x + width // 4, y + height // 4))

    if selected_cells:
        overlay = pygame.Surface((screen.get_width(), screen.get_height()), pygame.SRCALPHA)
        for row, col in selected_cells:
            x, y, width, height = col * cell_size, row * cell_size, cell_size, cell_size
            pygame.draw.rect(overlay, GREEN, (x, y, width, height), 0)
        screen.blit(overlay, (0,0))

    pygame.display.flip()

# Function to display traced words with a right-aligned title
def draw_word_list(screen, traced_words, screen_width, scroll_offset, attempt_counter):
    font_size = 36
    title_font_size = 40
    margin = 20

    traced_words_list = list(traced_words)
    title_text = pygame.font.Font(None, title_font_size).render(f"Completed Words - Attempts: {attempt_counter}", True, BLACK)
    title_width, title_height = title_text.get_size()
    title_x = screen_width - title_width
    screen.blit(title_text, (title_x,0))

    total_visible_words = min(25, len(traced_words_list))
    visible_word_count = min(total_visible_words, len(traced_words_list) - scroll_offset)

    for i in range(scroll_offset, scroll_offset + visible_word_count):
        if 0 <= i < len(traced_words_list):
            word = traced_words_list[i]
            text = pygame.font.Font(None, font_size).render(word, True, BLACK)
            x = screen_width - title_width
            y = (i) * (text.get_height() + margin) + title_height
            screen.blit(text, (x,y))

# Function to draw the attempted path in red
def draw_attempted_path(screen, path, cell_size):
    path_color = (255,0,0)  # Red color for attempted paths

    for row, col in path:
        x, y, width, height = col * cell_size, row * cell_size, cell_size, cell_size
        pygame.draw.rect(screen, path_color, (x, y, width, height),0)

        pygame.display.flip()

# Function to draw the AI's attempted path in blue
def draw_ai_attempted_path(screen, ai_path, cell_size):
    path_color = (0,0,255)  # Blue color for AI attempted paths

    for row, col in ai_path:
        x, y, width, height = col * cell_size, row * cell_size, cell_size, cell_size
        pygame.draw.rect(screen, path_color, (x, y, width, height),0)

        pygame.display.flip()

# Function to find English words and their paths in the grid using trie and NLP
def find_english_words_with_paths(grid, trie):
    words_with_paths = {}
    grid_size = len(grid)

    def is_valid_coord(row, col):
        return 0 <= row < grid_size and 0 <= col < grid_size

    def search_from_position(row, col, direction):
        path = []
        word = ""
        node = trie.root
        while is_valid_coord(row, col):
            path.append((row, col))
            word += grid[row][col]
            if grid[row][col] not in node.children:
                break
            node = node.children[grid[row][col]]
            if node.is_end_of_word and len(word) >1 and is_valid_word(word) and is_straight_line(path):
                words_with_paths[word] = path.copy()
            row, col = row + direction[0], col + direction[1]

    def is_straight_line(path):
        if len(path) <2:
            return True  # A single cell is considered a straight line

        # Check if all the cells are in the same row
        if all(row == path[0][0] for row, _ in path):
            return True

        # Check if all the cells are in the same column
        if all(col == path[0][1] for _, col in path):
            return True

        # Check if the line is diagonal
        dx = path[-1][0] - path[0][0]
        dy = path[-1][1] - path[0][1]
        if abs(dx) == abs(dy):
            return True

        # Otherwise, the line is not straight
        return False

    for row in range(grid_size):
        for col in range(grid_size):
            search_from_position(row, col, (0,1))  # Search horizontally
            search_from_position(row, col, (1,0))  # Search vertically
            search_from_position(row, col, (-1,1))  # Search diagonally (up-right)
            search_from_position(row, col, (1,1))  # Search diagonally (down-right)
            search_from_position(row, col, (-1,-1))  # Search diagonally (down-left)
            search_from_position(row, col, (0,-1))  # Search horizontally (left)
            search_from_position(row, col, (-1,0))  # Search vertically (up)
            search_from_position(row, col, (1,-1))  # Search diagonally (down-left)

    return words_with_paths

# Function to check if a word is valid using NLP
def is_valid_word(word):
    input_ids_bert = bert_tokenizer.encode(word, return_tensors='tf')
    outputs_bert = bert_model.predict(input_ids_bert)
    predicted_class_bert = tf.argmax(outputs_bert['logits'], axis=1).numpy().item()

    # BERT check: If BERT predicts the word as valid, return True
    if predicted_class_bert == 1:
        predicted_word = bert_tokenizer.decode(input_ids_bert[0])
        output_string = predicted_word.replace("[CLS]", "").replace("[SEP]", "").replace(" ", "")
        print(f"outuput_string_bert: {output_string}")

        input_ids_gpt2 = gpt2_tokenizer.encode(output_string, return_tensors='tf')
        outputs_gpt2 = gpt2_model.predict(input_ids_gpt2)
        predicted_class_gpt2 = tf.argmax(outputs_gpt2[0], axis=1).numpy()
        if predicted_class_gpt2.size > 1:
            # Handle the case where there are multiple elements along the axis
            # You may want to choose a specific element or apply a different logic here
            print("Warning: Multiple elements along the specified axis.")
            predicted_class_gpt2 = predicted_class_gpt2[0]
        else:
            predicted_class_gpt2 = predicted_class_gpt2.item()

        if (predicted_class_gpt2 == 1).any():
            predicted_word_gpt2 = gpt2_tokenizer.decode(input_ids_gpt2[0])
            print(f"output_string_gpt2: {predicted_word_gpt2}")
        
            # NLP checks for additional validation
            doc = nlp(predicted_word_gpt2)

            # Check if the word is a noun, verb, adjective, or adverb
            valid_pos = {'NOUN', 'VERB'}
            if any(token.pos_ in valid_pos for token in doc):
                return True

            # Check if the word is a proper noun
            if any(ent.label_ == 'PROPN' or ent.label_ == 'PROPV' for ent in doc.ents):
                return True

            # Fuzzy matching for approximate string matching
            for dictionary_word in nltk_words.words():
                if len(dictionary_word) >= 4 and fuzz.ratio(output_string, dictionary_word) >= 70:
                    return True

            # If none of the conditions is met, return False
            return False

# Function to solve the word search puzzle using trie and NLP
def solve_puzzle(grid, trie, event, attempt_counter, lock, traced_words):
    words_with_paths = {}
    directions = [(0,1), (1,0), (-1,1), (1,1), (-1,-1), (0,-1), (-1,0), (1,-1)]

    def is_valid_coord(row, col):
        return 0 <= row < len(grid) and 0 <= col < len(grid[0])

    def grid_to_word(path, grid):
        return ''.join([grid[row][col] for row, col in path])

    def backtrack(row, col, path, node, visited):
        nonlocal attempt_counter
        if not is_valid_coord(row, col) or visited[row][col]:
            return

        letter = grid[row][col]
        if letter not in node.children:
            return

        node = node.children[letter]
        path.append((row, col))

        if node.is_end_of_word and len(path) >1 and is_valid_word(grid_to_word(path, grid)) and is_straight_line(path) and len(grid_to_word(path, grid)) >=4:
            with lock:
                traced_words.add(grid_to_word(path, grid))
                attempt_counter +=1  # Increment attempt counter

        visited[row][col] = True

        for direction in directions:
            new_row, new_col = row + direction[0], col + direction[1]
            backtrack(new_row, new_col, path, node, visited)

        visited[row][col] = False
        path.pop()

    def is_straight_line(path):
        if len(path) <2:
            return True  # A single cell is considered a straight line

        # Check if all the cells are in the same row
        if all(row == path[0][0] for row, _ in path):
            return True

        # Check if all the cells are in the same column
        if all(col == path[0][1] for _, col in path):
            return True

        # Check if the line is diagonal
        dx = path[-1][0] - path[0][0]
        dy = path[-1][1] - path[0][1]
        if abs(dx) == abs(dy):
            return True

        # Otherwise, the line is not straight
        return False

    ai_path = []  # Store AI's attempted path

    for row in range(len(grid)):
        for col in range(len(grid[0])):
            visited = [[False] * len(grid[0]) for _ in range(len(grid))]
            current_path = []  # Store the current path for each starting position
            backtrack(row, col, current_path, trie.root, visited)

            # Store the AI's path
            with lock:
                ai_path.extend(current_path)

    event.set()  # Notify the main thread that the solving is complete
    return words_with_paths, ai_path

class SolverThread(Thread):
    def __init__(self, screen, cell_size, grid, trie, event, callback, attempt_counter, lock, traced_words):
        super().__init__()
        self.screen = screen
        self.cell_size = cell_size
        self.grid = grid
        self.trie = trie
        self.event = event
        self.callback = callback
        self.attempt_counter = attempt_counter
        self.lock = lock
        self.traced_words = traced_words

    def run(self):
        result = solve_puzzle(self.grid, self.trie, self.event, self.attempt_counter, self.lock, self.traced_words)
        pygame.event.post(pygame.event.Event(AI_UPDATE_EVENT, ai_path=result[1]))

def get_grid_size():
    root = Tk()
    root.withdraw()
    size = simpledialog.askinteger("Grid size", "Enter size of map: ")
    return size

def update_ai_path(result, lock, traced_words, screen, cell_size):
    words_with_paths, ai_path = result
    traced_words.update(words_with_paths.keys())
    draw_ai_attempted_path(screen, ai_path, cell_size)
    pygame.display.flip()

def main():
    pygame.font.init()

    grid_size = get_grid_size()
    english_word_percentage = 10

    cell_size = max(20, min(800 // grid_size,800 // grid_size))
    width, height = grid_size * cell_size, grid_size * cell_size

    screen = pygame.display.set_mode((width +600, height))
    pygame.display.set_caption("Word Search Puzzle AI - Attempts:0")

    grid = generate_grid(grid_size, english_word_percentage)

    trie = Trie()
    for word in nltk_words.words():
        if 4 <= len(word) <=8:
            trie.insert(word.upper())

    running = True
    title_visible = True
    selected_cells = set()
    traced_words = set()
    solving_event = Event()
    solving_thread = None
    attempt_counter =0
    lock = Lock()

    clock = pygame.time.Clock()
    scroll_offset =0

    AI_UPDATE_EVENT = pygame.USEREVENT +1

    # Load the neural network model

    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == pygame.USEREVENT:
                ai_path = event.dict.get("ai_path")
                if ai_path:
                    with lock:
                        draw_ai_attempted_path(screen, ai_path, cell_size)
                        pygame.display.flip()

        if solving_thread and solving_event.is_set():
            solving_thread.join()
            solving_thread = None
            solving_event.clear()

        screen.fill(WHITE)
        draw_word_list(screen, traced_words, width +450, scroll_offset, attempt_counter)
        draw_grid(screen, grid, cell_size, selected_cells)

        solving_thread = SolverThread(screen, cell_size, grid, trie, solving_event, update_ai_path, attempt_counter, lock, traced_words)

        if solving_thread and solving_thread.is_alive() == False:
            solving_thread.start()
        else:
            print("Already run")

        pygame.display.flip()

        clock.tick(120)

    pygame.quit()

if __name__ == "__main__":
    main()
