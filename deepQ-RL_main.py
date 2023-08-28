from flappy_bird_functions import *
from neural_network import *
from deep_Q import run_game
import copy
import pickle

def take_action(Q_nn, state_input, game_idx, epsilon=0.15):
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
n_neurons_array = [20] * n_hidden_layers
Q_nn = NeuralNetwork(input_length, output_length, n_hidden_layers, n_neurons_array, learning_rate=5e-4 ,activation=ReLU())
Q_hat_nn = copy.deepcopy(Q_nn)

n_games = 250
D_buffer = [None] * 50000   # replay buffer
global_D_idx = 0
hi_score = 0

all_cum_rewards = [None] * n_games
all_scores  = [None] * n_games

for game_idx in range(n_games):

    Q_nn, Q_hat_nn, D_buffer, global_D_idx, score, cum_reward = run_game(take_action, Q_nn, Q_hat_nn, D_buffer, global_D_idx, game_idx)
    if score>hi_score:
        hi_score = copy.deepcopy(score)
    print(f"Game {game_idx} run - cum_reward {cum_reward} -  score {score} - current record: {hi_score}")
    
    all_cum_rewards[game_idx] = cum_reward
    all_scores[game_idx] = score

with open('final_NN', 'wb') as file: pickle.dump(Q_nn, file)
np.savetxt('reward_curve', all_cum_rewards)
np.savetxt('score_curve', all_scores)

