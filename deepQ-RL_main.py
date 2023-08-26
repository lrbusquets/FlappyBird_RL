from flappy_bird_functions import *
from neural_network import *
from deep_Q import run_game
import copy

def take_action(Q_nn, state_input, game_idx, epsilon=0.1):
    state = copy.deepcopy(state_input)
    epsilon = epsilon * np.exp(-game_idx/10000*4)

    if np.random.random() < epsilon:
        print(f"Taking action: random")
        return 1.0 if np.random.random() > 0.9 else 0.0
    else:
        state.append(0.0)
        q0 = Q_nn.evaluate(state)
        state[-1] = 1.0
        q1 = Q_nn.evaluate(state)
        print(f"Taking action: q0={q0} vs q1={q1} --> {q1>q0}")

        return 1.0 if q1 > q0 else 0.0


input_length = 8  # state + action + action_prev
output_length = 1
n_hidden_layers = 5
n_neurons_array = [5] * n_hidden_layers
Q_nn = NeuralNetwork(input_length, output_length, n_hidden_layers, n_neurons_array, learning_rate=5e-4 ,activation=ReLU())
Q_hat_nn = copy.deepcopy(Q_nn)

n_games = 25000
D_buffer = [None] * 50000   # replay buffer
global_D_idx = 0
hi_score = 0

for game_idx in range(n_games):

    Q_nn, Q_hat_nn, D_buffer, global_D_idx, score, cum_reward = run_game(take_action, Q_nn, Q_hat_nn, D_buffer, global_D_idx, game_idx)
    if score>hi_score:
        hi_score = copy.deepcopy(score)
    print(f"Game run - cum_reward {cum_reward} -  score {score} - current record: {hi_score}")

