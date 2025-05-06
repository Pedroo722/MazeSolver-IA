import pygame
import numpy as np
import random
import time

# Configurações
GRID_SIZE = 8
CELL_SIZE = 50
WIDTH = HEIGHT = GRID_SIZE * CELL_SIZE
START = (0, 0)
GOAL = (7, 7)
OBSTACLES = [
    (6, 1), (5, 1), (2, 1), (3, 1),
    (2, 3), (3, 3), (4, 4), (5, 4),
    (6, 5), (1, 6), (2, 6), (3, 7), (5, 7)]

ACTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # cima, baixo, esquerda, direita

# Parâmetros do Q-Learning
ALPHA = 0.1
GAMMA = 0.9
EPSILON = 0.2
EPISODES = 500
MAX_STEPS = 300

# Cores
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREY = (200, 200, 200)
BLUE = (50, 50, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
PURPLE = (128, 0, 128)
ORANGE = (255, 165, 0)
LIGHT_BLUE = (173, 216, 230)

# Q-table
q_table = np.zeros((GRID_SIZE, GRID_SIZE, len(ACTIONS)))

# Variáveis para o triângulo laranja e o estado do agente
triangle_pos = None
has_key = False  # Indica se o agente pegou a chave

# Funções auxiliares
def is_valid(pos):
    x, y = pos
    return 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE and pos not in OBSTACLES

def get_next_state(state, action):
    dx, dy = ACTIONS[action]
    next_state = (state[0] + dx, state[1] + dy)
    # Verifica se o próximo estado é válido e se o agente pode passar
    if is_valid(next_state):
        if next_state in yellow_squares and not has_key:
            return state  # Se o agente não tem a chave, ele não pode passar
        return next_state
    return state

def get_reward(state):
    global has_key
    if state == GOAL:
        print("O Agente alcançou o objetivo!")
        return 5000  # Recompensa por alcançar o objetivo
    elif state == triangle_pos:
        if not has_key:
            has_key = True
            return 100  # Recompensa por pegar a chave
        else:
            return -5  # Penalidade por voltar à casa da chave após já tê-la pegado
    elif state in yellow_squares and not has_key:
        return -20  # Penalidade por tentar passar pelos quadrados amarelos sem a chave
    elif state in OBSTACLES:
        return -20  # Penalidade por bater em um obstáculo
    else:
        return -1  # Penalidade por movimento comum

def draw_grid(screen):
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            color = WHITE
            if (x, y) in OBSTACLES:
                color = BLACK
            elif (x, y) in yellow_squares:  # Quadrados amarelos ao redor do objetivo, precisam de chaver para passar
                color = YELLOW
            elif (x, y) == START:
                color = GREEN
            elif (x, y) == GOAL:
                color = RED
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, GREY, rect, 2)

def draw_agent(screen, pos, color=BLUE):
    rect = pygame.Rect(pos[0] * CELL_SIZE + 5, pos[1] * CELL_SIZE + 5, CELL_SIZE - 10, CELL_SIZE - 10)
    pygame.draw.ellipse(screen, color, rect)

def draw_triangle(screen, pos):
    points = [
        (pos[0] * CELL_SIZE + CELL_SIZE // 2, pos[1] * CELL_SIZE + 5),
        (pos[0] * CELL_SIZE + 5, pos[1] * CELL_SIZE + CELL_SIZE - 5),
        (pos[0] * CELL_SIZE + CELL_SIZE - 5, pos[1] * CELL_SIZE + CELL_SIZE - 5)
    ]
    pygame.draw.polygon(screen, ORANGE, points)

def process_events():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

# Função para definir os quadrados amarelos ao redor do objetivo
def set_yellow_squares():
    global yellow_squares
    yellow_squares = set()
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if (GOAL[0] + dx, GOAL[1] + dy) != GOAL and 0 <= GOAL[0] + dx < GRID_SIZE and 0 <= GOAL[1] + dy < GRID_SIZE:
                yellow_squares.add((GOAL[0] + dx, GOAL[1] + dy))

# -----------------------------
# TREINAMENTO COM VISUALIZAÇÃO
# -----------------------------
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Treinamento do Agente")
clock = pygame.time.Clock()

set_yellow_squares()

for episode in range(EPISODES):
    print(f"Treinando episódio {episode + 1}/{EPISODES}")

    while True:
        triangle_pos = (6, 0)
        if triangle_pos not in OBSTACLES:
            break
    
    state = START
    has_key = False  # O agente não tem a chave no início do episódio

    for step in range(MAX_STEPS):
        process_events()

        if random.random() < EPSILON:
            action = random.randint(0, 3)
        else:
            action = np.argmax(q_table[state[0], state[1]])

        next_state = get_next_state(state, action)
        reward = get_reward(next_state)

        old_value = q_table[state[0], state[1], action]
        next_max = np.max(q_table[next_state[0], next_state[1]])
        q_table[state[0], state[1], action] = old_value + ALPHA * (reward + GAMMA * next_max - old_value)

        state = next_state

        # Visualização do treinamento
        screen.fill(WHITE)
        draw_grid(screen)
        draw_agent(screen, state, YELLOW)
        if not has_key:
            draw_triangle(screen, triangle_pos)
        pygame.display.flip()
        clock.tick(60)

        if state == GOAL:
            break

print("Treinamento concluído!")
time.sleep(1)
pygame.display.quit()

# -----------------------------
# EXECUÇÃO DO AGENTE TREINADO
# -----------------------------
pygame.display.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Execução do Agente Treinado")
clock = pygame.time.Clock()

agent_pos = START
path = [agent_pos]
reached_goal = False
running = True

while running:
    process_events()

    screen.fill(WHITE)
    draw_grid(screen)

    # Desenha caminho já percorrido
    for pos in path:
        draw_agent(screen, pos, BLUE)

    # Desenha agente atual
    if not reached_goal:
        draw_agent(screen, agent_pos, PURPLE)

    if not has_key:
        draw_triangle(screen, triangle_pos)

    pygame.display.flip()
    clock.tick(30)  # FPS alto para manter janela fluida

    if not reached_goal:
        if agent_pos != GOAL:
            action = np.argmax(q_table[agent_pos[0], agent_pos[1]])
            next_pos = get_next_state(agent_pos, action)
            if next_pos == agent_pos:
                print("Agente está preso!")
                reached_goal = True
            else:
                path.append(next_pos)
                agent_pos = next_pos
                time.sleep(0.8)  # Controlando a velocidade da execução
        else:
            reached_goal = True
            print("\nCaminho percorrido pelo agente:")
            print(path)