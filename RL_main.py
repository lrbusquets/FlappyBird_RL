from flappy_bird_functions import *
from neural_network import *
from ACD import run_game


def policy_function(actor, state):
    #return 1 if random.uniform(0, 1) > 0.95 else 0
    return 1 if actor.evaluate(state) > actor_boundary else 0

def get_error_critic(r,J):
    
    assert len(r) == len(J), "The inputted reward and critic value arrays don't have the same length"
    n = len(J) - 1
    ec = [None] * n 
    Ec = np.zeros(n)
    for i in range(n):
        ec_val = J[i+1] - (gamma * J[i] - r[i])
        ec[i] = [ec_val]
        Ec[i] = 0.5 * (ec_val)**2

    return Ec, np.array(ec)

input_length = 7  # state + action_prev
output_length = 1
n_hidden_layers = 8
n_neurons_array = [8] * n_hidden_layers
beta = 1e-5
actor = NeuralNetwork(input_length, output_length, n_hidden_layers, n_neurons_array, learning_rate=beta, activation=Sigmoid())

input_length = 6 + 2  # state + action + action_prev
output_length = 1
n_hidden_layers = 8
n_neurons_array = [8] * n_hidden_layers
beta = 1e-5
critic = NeuralNetwork(input_length, output_length, n_hidden_layers, n_neurons_array, learning_rate=beta, activation=ReLU())

n_games = 1000
#n_fwdback_iterations = 20

for _ in range(n_games):

    cum_reward, all_rewards, J_values, critic, actor = run_game(policy = policy_function, critic=critic, actor=actor)

    #critic = critic_updated
    #actor = actor_updated
    #Ec, ec = get_error_critic(all_rewards, J_values)
    #print(ec)
    #print(f"RMS of Ec = {rms(Ec)}")
    #critic.backprop(-gamma*ec)

#cum_reward, all_rewards, J_values = run_game(policy = policy_function, critic=critic)
#Ec, ec = get_error_critic(all_rewards, J_values)
#print(f"RMS of Ec = {rms(Ec)}")
#critic.backprop(-gamma*ec)