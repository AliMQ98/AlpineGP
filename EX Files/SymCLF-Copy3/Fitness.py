# Fitness Function
import ray
from SymFunctions import get_features_batch, compile_individuals
from Evaluate import eval_MSE_sol

from stable_or_not import stable_or_not

@ray.remote
def fitness(individuals_str, toolbox, true_data, penalty):

    MSEE = stable_or_not(individuals_str)
    
    callables = compile_individuals(toolbox, individuals_str)

    individ_length, nested_trigs, num_trigs = get_features_batch(individuals_str)
    MSE = [None] * len(individuals_str)
    fitnesses = [None] * len(individuals_str)
    for i, ind in enumerate(callables):
        if individ_length[i] >= 300:
            fitnesses[i] = (1e8,)
        else:
            MSE[i], _ = eval_MSE_sol(ind, true_data)

            fitnesses[i] = (
                MSE[i]
                + 100000 * nested_trigs[i]
                + penalty["reg_param"] * individ_length[i],
            )
            if MSEE[i] > 10e9:
                MSE[i] = MSE[i] + MSEE[i]
    print(MSE)
            
    # Find the index of the minimum fitness
    min_index = fitnesses.index(min(fitnesses))
    # Print the minimum fitness and the corresponding individual
    #print(f"Min Fitness: {fitnesses[min_index]}, Corresponding Individual: {individuals_str[min_index]}, MSE: {MSE[min_index]}")
    return fitnesses