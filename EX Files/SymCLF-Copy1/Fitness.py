# Fitness Function
import ray
from SymFunctions import get_features_batch, compile_individuals
from Evaluate import eval_MSE_sol

@ray.remote
def fitness(individuals_str, toolbox, true_data, penalty):
    callables = compile_individuals(toolbox, individuals_str)

    individ_length, nested_trigs, num_trigs = get_features_batch(individuals_str)

    fitnesses = [None] * len(individuals_str)
    for i, ind in enumerate(callables):
        if individ_length[i] >= 50:
            fitnesses[i] = (1e8,)
        else:
            MSE, _ = eval_MSE_sol(ind, true_data)

            fitnesses[i] = (
                MSE
                + 100000 * nested_trigs[i]
                + penalty["reg_param"] * individ_length[i],
            )

    return fitnesses