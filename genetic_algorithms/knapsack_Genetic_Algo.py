import numpy as np
import random as rd
from random import randint


def print_list(weight, value):
    print('List of articles')
    print('Number   Weight   Value')
    for i in range(len(weight)):
        print('{0}\t{1}\t{2}'.format(i, weight[i], value[i]))


#the higher the value we can add to the backpack without excceding max weight the better
def fitness_function(weight, value, population, threshold):
    fitness = np.empty(population.shape[0]) #number of individuals
    for i in range(population.shape[0]):
        S1 = np.sum(population[i] * value)
        S2 = np.sum(population[i] * weight) 
        if S2 <= threshold:
            fitness[i] = S1 
        else :
            fitness[i] = 0 
    return fitness.astype(int)



def selection(fitness, num_parents, population):
    fitness = list(fitness) 
    parents = np.empty((num_parents, population.shape[1]))
    for i in range(num_parents):
        max_fitness_idx = np.where(fitness == np.max(fitness))
        parents[i,:] = population[max_fitness_idx[0][0], :]
        fitness[max_fitness_idx[0][0]] = -999999
    return parents

def crossover(parents, num_offsprings):
    offsprings = np.empty((num_offsprings, parents.shape[1]))
    crossover_point = int(parents.shape[1]/2)
    crossover_rate = 0.8
    i=0
    while (parents.shape[0] < num_offsprings):
        parent1_index = i%parents.shape[0]
        parent2_index = (i+1)%parents.shape[0]
        x = rd.random()
        if x > crossover_rate:
            continue
        parent1_index = i%parents.shape[0]
        parent2_index = (i+1)%parents.shape[0]
        offsprings[i,0:crossover_point] = parents[parent1_index,0:crossover_point]
        offsprings[i,crossover_point:] = parents[parent2_index,crossover_point:]
        i=+1
    return offsprings    


def mutation(offsprings):
    mutants = np.empty((offsprings.shape))
    mutation_rate = 0.5
    for i in range(mutants.shape[0]):
        random_value = rd.random()
        mutants[i,:] = offsprings[i,:]
        if random_value > mutation_rate:
            continue
        int_random_value = randint(0,offsprings.shape[1]-1)    
        if mutants[i,int_random_value] == 0 :
            mutants[i,int_random_value] = 1
        else :
            mutants[i,int_random_value] = 0
    return mutants   


def optimize(weights, values, population, pop_size, num_gen, knapsack_threshold):
    parameters = []
    num_parents = int(pop_size[0]/2)
    num_offsprings = pop_size[0] - num_parents 
    for i in range(num_gen):
        fitness = fitness_function(weights, values, population, knapsack_threshold)
        parents = selection(fitness, num_parents, population)
        offsprings = crossover(parents, num_offsprings)
        mutants = mutation(offsprings)
        population[0:parents.shape[0], :] = parents
        population[parents.shape[0]:, :] = mutants
        
    fitness_last_gen = fitness_function(weights, values, population, knapsack_threshold)
    max_fitness = np.where(fitness_last_gen == np.max(fitness_last_gen))

    parameters.append(population[max_fitness[0][0],:])
    return parameters, np.max(fitness_last_gen)



number_items = 10

#weight = np.random.randint(1, 15, size = number_items)#create 10 weight with weight in [1,15)
#value = np.random.randint(10, 750, size = number_items)#create 10 values with values in [10,750)
weights = [ 7,  4,  4,  8, 12,  7,  5,  7, 11,  7]
values = [505, 505, 122, 395, 367, 191, 624, 737, 248, 606]


knapsack_threshold = 35    #Maximum weight that the bag can have inside
chromosomes = 64 #number of individuals(chromosomes) per  population (it's a power of two)
num_gen = 50 #number of generations 



pop_size = (chromosomes, number_items)# each individual has fixed length
initial_population = np.random.randint(2, size = pop_size) #randomly creation of individuals in [2,) 
initial_population = initial_population.astype(int) #Copy of the array, cast to a specified type.


parameters,maximum = optimize(weights, values, initial_population, pop_size, num_gen, knapsack_threshold)


print_list(weights, values)


print('\nSelected items that will maximize the knapsack:')
for i in range(len(parameters[0])):
    if parameters[0][i]:
        print('{}\n'.format(i))

print('Max:{}'.format(maximum))

