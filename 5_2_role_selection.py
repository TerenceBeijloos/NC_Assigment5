import numpy as np
import matplotlib.pyplot as plt

def generate_bit_sequence(length):
    return np.random.randint(2, size=length)

def mutate_bit_sequence(sequence, mutation_rate):
    mutated_sequence = sequence.copy()
    for i in range(len(mutated_sequence)):
        if np.random.rand() < mutation_rate:
            mutated_sequence[i] = 1 - mutated_sequence[i]
    return mutated_sequence

def fitness(length, sequence):
    fitness = 0
    for i in range(len(sequence)):
        fitness += sequence[i]*2**(length-i-1)
    return fitness

def ones_ga(length, mutation_rate, generations):
    best_fitnesses = []
    
    x = generate_bit_sequence(length)
    best_fitness = fitness(length, x)
    best_fitnesses.append(best_fitness)

    for i in range(generations):
        xm = mutate_bit_sequence(x, mutation_rate)
        # if fitness(length, xm) > best_fitness:
        x = xm
        best_fitness = fitness(length, x)
        best_fitnesses.append(best_fitness)

        # if best_fitness >= (2**length)-1:
        #     break  

    return best_fitnesses, i

def write_to_file(data, filename):
    with open(filename, 'w') as f:
        for i in data:
            f.write(f"{i}\n")

def read_from_file(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            fitness = float(line.strip()) 
            data.append(fitness)
    return data

if __name__ == "__main__":
    l = 100
    mutation_rate = 1 / l
    generations = 1500

    random_data, generations = ones_ga(l, mutation_rate, generations)
    write_to_file(random_data, "random.txt")

    best_fitness_data = read_from_file("best_fitness.txt")

    performance_difference = [best_fitness_data[i] - random_data[i] for i in range(len(random_data))]

    mean_performance_difference = np.mean(performance_difference)
    std_dev_performance_difference = np.std(performance_difference)

    generations = np.arange(len(random_data))

    plt.plot(generations, performance_difference, label='Performance Difference', color='b')

    plt.axhline(y=0, color='k', linestyle='--')

    plt.xlabel('Generation')
    plt.ylabel('Performance Difference')
    plt.title('Performance Difference Between Best Fitness and Random Selection')
    plt.legend()

    plt.xlim(generations.min(), generations.max())
    plt.ylim(min(performance_difference), max(performance_difference))
    
    plt.show()







