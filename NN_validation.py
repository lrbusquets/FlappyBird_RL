from neural_network import *
import matplotlib.pyplot as plt

fun="sinc"

if fun == "sqrt" or fun == "sinc":
    input_length = 1
elif fun == "ripple":
    input_length = 2

def function_plot(my_NN):

    if input_length == 1:
        # generate the two curves to compare
        n = 1000
        x_array = np.linspace(-3,3,n)
        y_array = my_fun(x_array, fun=fun)
        y_NN_array = np.empty(n)
        for i in range(n):
            y_NN_array[i] = my_NN.evaluate([x_array[i]])[0]

        plt.figure(figsize=(8, 6))  # Optional: Set the figure size

        # Plot the curves
        plt.plot(x_array, y_array, label='function')
        plt.plot(x_array, y_NN_array, label='NN approximation')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()  # Display legend
        plt.grid(True)  # Optional: Add grid
        plt.show()

    elif input_length == 2:

        # Create a meshgrid for the coordinates
        n = 200
        x = np.linspace(-5, 5, n)
        y = np.linspace(-5, 5, n)


        X, Y = np.meshgrid(x, y)
        #Z = ripple_function(X, Y, 0, frequency=2, amplitude=0.5)
        z = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                z[i, j] = my_NN.evaluate([x[i], y[j]])
                #z[i, j] = my_fun([x[i], y[j]], fun=fun)

        # Create a 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the surface
        surf = ax.plot_surface(X, Y, z, cmap='viridis')

        # Add labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        #ax.set_title('3D Surface with Ripples')

        # Add a color bar
        #fig.colorbar(surf)

        # Show the plot
        plt.show()


def my_fun(input, fun="sqrt"):
    if fun == "sqrt":
        return np.sqrt(np.abs(input))
    elif fun == "sinc":
        return 0.075 * np.sin(10*input)/input + 0.2
    elif fun == "ripple":
        freq = 2
        amp = 0.5
        x = input[0]
        y = input[1]
        r = np.sqrt(x**2 + y**2)
        return amp * np.cos(freq * r) * np.exp(-0.5*r)

    
def get_dataset(n_samples=1000, method="random"):
    
    if method == "random":
        inputs_array = np.array([None] * n_samples)
        outputs_array = np.array([None] * n_samples)
        for i in range(n_samples):
            inputs_array[i] = np.random.randn(input_length) * 3
            outputs_array[i] = my_fun(inputs_array[i], fun=fun)

    elif method == "uniform":
        inputs_array = np.linspace(-3,3,n_samples)
        inputs_array = inputs_array[:, np.newaxis]
        outputs_array = np.array([None] * n_samples)
        for i in range(n_samples):
            outputs_array[i] = my_fun(inputs_array[i], fun=fun)

    return inputs_array, outputs_array



output_length = 1
n_hidden_layers = 3
n_neurons_array = [6] * n_hidden_layers
beta = 0.02
my_NN = NeuralNetwork(input_length, output_length, n_hidden_layers, n_neurons_array, learning_rate=beta, activation=ReLU())

n_epochs = int(6e2)
MSE_array = [None] * n_epochs

# Enable interactive mode for live updating
plt.ion()

# Create the initial plot
plt.figure(figsize=(8, 6))
line, = plt.plot([], [], label='Training error')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.title('Training error Over Time')
plt.legend()
plt.grid(True)

n_samples = 500
inputs_array, outputs_array = get_dataset(n_samples)


for t in range(n_epochs):

    MSE_val, std = my_NN.train(inputs_array, outputs_array, training_it=t)

    #NN_outputs = np.array([None] * n_samples)
    #for i in range(n_samples):
    #    NN_outputs[i] = my_NN.evaluate(inputs_array[i])

    #error = NN_outputs - outputs_array
    #MSE_val = mse(error)
    #std = np.std(error)
    MSE_array[t] = MSE_val

    #print(f"Epoch {t}, input0={inputs_array[0]}, output0={NN_outputs[0]} MSE={MSE_val}, W000={my_NN.W_matrices[0][0][0]}, b00={my_NN.b_vectors[0][0]}")
    print(f"Epoch {t},  MSE={MSE_val}, W000={my_NN.W_matrices[0][0][0]}, b00={my_NN.b_vectors[0][0]}")
    #function_plot(my_NN)

    #my_NN.backprop(2*error)
    #my_NN.learning_rate = beta * np.exp(-0.01*t)

    
    epochs = np.linspace(0,t,t+1)
    line.set_xdata(epochs)
    error2plot = MSE_array[:(t+1)]
    line.set_ydata(error2plot)
    plt.xlim(0, t - 1)
    plt.ylim(0.1 * np.min(error2plot), 10 * np.max(error2plot))
    plt.yscale('log')
    plt.fill_between([t], MSE_array[t] - std, MSE_array[t] + std, color='gray', alpha=0.2)
    plt.pause(0.01)

    if t>0 and np.abs(MSE_array[t]/MSE_array[t-1] - 1) < (1e-4 * MSE_array[0]):
        inputs_array, outputs_array = get_dataset(n_samples)
        print("Updating dataset...")

plt.ioff()
plt.show()

# save learning curve
np.savetxt('data_files/benchmark_1_MSE_array.dat', np.array(MSE_array))

function_plot(my_NN)

input("Press enter to exit")



#print(inputs_array)
#print(outputs_array)

