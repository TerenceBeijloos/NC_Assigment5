import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

target_string = "GeneticAlgorithm"
A = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
population_size = 200
K = 2
mutation_rates = [0, 3 / len(target_string), 1 / len(target_string)]  # µ = 0, µ = 1/L and µ = 3/L
crossover_probability = 1
Gmax = 250

def hamming_distance(string1, string2): 
    distance = 0

    for i in range(len(string1)):
        if string1[i] != string2[i]:
            distance += 1
            
    return distance
def average_hamming_distance(population):
    distances = []
    n = population_size

    for i in range(n):
        for j in range(i + 1, n):
            distances.append(hamming_distance(population[i], population[j]))

    return np.mean(distances) if distances else 0

def generate_sequence(length):
    return [A[random.randint(0, (len(A)-1))] for _ in range(length)]

def generate_population(population_size, sequence_length):
    return [generate_sequence(sequence_length) for _ in range(population_size)]

def fitness(sequence, goal):
    matches = 0
    for i in range(len(sequence)):
        if sequence[i] == goal[i]:
            matches += 1
    return matches / len(sequence)

def tournament_selection(population, fitnesses, K):
    selected = []
    for _ in range(2):
        participants_indices = np.random.choice(population_size, size=K, replace=False)
        participants_fitnesses = [fitnesses[i] for i in participants_indices]
        winner_index = participants_indices[np.argmax(participants_fitnesses)]
        selected.append(population[winner_index])
    return selected

def crossover(parent1, parent2):
    if np.random.rand() < crossover_probability:
        crossover_point = np.random.randint(1, len(parent1))
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        return child1, child2
    else:
        return parent1, parent2

def mutate(string, mutation_rate):
    mutated_string = ''
    for char in string:
        if np.random.rand() < mutation_rate:
            mutated_string += np.random.choice([c for c in A if c != char])
        else:
            mutated_string += char
    return mutated_string

def generate_offspring(population, fitnesses, mutation_rate):
    offspring = []
    for _ in range(population_size // 2):
        parent1, parent2 = tournament_selection(population, fitnesses, K)
        child1, child2 = crossover(parent1, parent2)
        child1 = mutate(child1, mutation_rate)
        child2 = mutate(child2, mutation_rate)
        offspring.extend([child1, child2])
    return offspring

def string_search(target, generations, mutation_rate):
    population = generate_population(population_size, len(target))
    distance = []
    found_solution = False
    index = 0
    for i in range(generations):
        fitnesses = [fitness(string, target) for string in population]
        if 1 in fitnesses:
            found_solution = True
            index = i
            # return i, distance
            
        if not found_solution:
            population = generate_offspring(population, fitnesses, mutation_rate)
        
        if i % 10 == 0:
            distance.append(average_hamming_distance(population))

    return index if found_solution else generations , distance

if __name__ == "__main__":
    results = {mu: [] for mu in mutation_rates}
    distances = {mu: [] for mu in mutation_rates}
    runs = 10
    
    for mu in mutation_rates:
        print("Mutation Rate: ", mu)
        for i in range(runs):
            print("Run: ", i)
            generations, distance = string_search(target_string, Gmax, mu)
            results[mu].append(generations)
            distances[mu].append(distance)

    generation_data = []

    for mu in results:
        for gen in results[mu]:
            generation_data.append((mu, gen))
        
    df_gen = pd.DataFrame(generation_data, columns=['Mutation Rate', 'Generations'])

    sns.swarmplot(x='Mutation Rate', y='Generations', data=df_gen)
    plt.title('Generations Needed to Find Target')
    plt.show()

    for mu, diversity_lists in distances.items():
        data_array = np.array(diversity_lists)
        means = np.mean(data_array, axis=0)
        generations = np.arange(10, len(means) * 10 + 1, 10)

        plt.plot(generations, means, marker='o', linestyle='-', label=f'Mu={mu}')

    plt.title('Hamming Mean Values Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Hamming Mean Value')

    if distances:
        first_key = next(iter(distances))
        generations = np.arange(10, Gmax + 1, 10)
        plt.xticks(generations)
        
    plt.grid(True)
    plt.legend()
    plt.show()