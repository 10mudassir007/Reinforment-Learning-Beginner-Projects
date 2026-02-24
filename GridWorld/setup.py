import random

grid_size = 10
start = (1,1)
goal = (9,9)
obstacles = [(1,2),(2,3)]

alpha=0.1
discount=0.6
epsilon=0.2
episodes=500
max_steps=50

def valid_state(state):
    x, y = state
    if x < 0 or x >= grid_size or y < 0 or y >= grid_size:
        return False
    return True

def get_reward(state):
    if state == goal:
        return 100
    elif state in obstacles:
        return -40
    else:
        return -1
    
actions = ['up', 'down', 'left', 'right']

def next_state(state, action):
    x, y = state
    if action == 'down':
        next_s = (x,y+1)
    elif action == 'left':
        next_s = (x-1,y)
    elif action == 'right':
        next_s = (x+1,y)
    elif action == 'up':
        next_s = (x,y-1)
    else:
        next_s = state

    if valid_state(next_s):
        return next_s
    else:
        return state

def choose_action(Q,state, epsilon):
    if random.random() < epsilon:
        return random.choice(actions)
    else:
        return max(Q[state], key=Q[state].get)
    
Q = {}

for x in range(grid_size):
    for y in range(grid_size):
        Q[(x,y)] = {a:0 for a in actions}

for ep in range(episodes):
    state = start 
    for step in range(max_steps):
        action = choose_action(Q,state, epsilon)
        next_s = next_state(state, action)
        reward = get_reward(next_s)

        best_next_action = max(Q[next_s], key=Q[next_s].get)
        Q[state][action] += alpha * (reward + discount * Q[next_s][best_next_action] - Q[state][action])

        state = next_s
        if state == goal:
            print(f"Episode {ep} reached goal in {step+1} steps")
            break