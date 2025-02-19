# Fitness Function
import ray
from SymFunctions import get_features_batch, compile_individuals
from Evaluate import eval_MSE_sol

@ray.remote
def fitness(individuals_str, toolbox, true_data, penalty):
    """
    Computes the fitness of a batch of individuals in a symbolic regression problem.
    
    Parameters:
    individuals_str : list of str
        List of individuals represented as string expressions.
    toolbox : deap.base.Toolbox
        DEAP toolbox for evolutionary computations.
    true_data : numpy.ndarray
        The dataset against which the individuals are evaluated.
    penalty : dict
        A dictionary containing penalty parameters for fitness calculation.

    Returns:
    list of tuple
        A list containing fitness values for each individual.
    """

    # Compile individuals into callable functions
    callables = compile_individuals(toolbox, individuals_str)

    # Extract features from the individuals
    individ_length, nested_trigs, num_trigs = get_features_batch(individuals_str)

    # Initialize lists to store MSE and fitness values
    MSE = [None] * len(individuals_str)
    fitnesses = [None] * len(individuals_str)

    # Compute fitness for each individual
    for i, ind in enumerate(callables):
        # If the individual's length exceeds _, assign a very high fitness (penalty)
        if individ_length[i] >= 80:
            fitnesses[i] = (1e8,)
        else:
            # Evaluate the Mean Squared Error (MSE) of the individual
            MSE[i], _ = eval_MSE_sol(ind, true_data)

            # Compute the fitness function with penalties for complexity
            fitnesses[i] = (
                MSE[i]
                + 100000 * nested_trigs[i]
                + penalty["reg_param"] * individ_length[i],
            )
            
    # Find the index of the minimum fitness
    #min_index = fitnesses.index(min(fitnesses))
    # Print the minimum fitness and the corresponding individual
    #print(f"Min Fitness: {fitnesses[min_index]}, Corresponding Individual: {individuals_str[min_index]}, MSE: {MSE[min_index]}")
    return fitnesses