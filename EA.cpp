#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <iterator>
#include <numeric>

using namespace std;

const int population_size = 200;
const int K = 2;
const double mutation_rate = 0.2;
const double crossover_probability = 1;

struct City {
    double x, y;
};

vector<City> load_city_coordinates(const string& filename) {
    vector<City> cities;
    ifstream file(filename);
    double x, y;
    while (file >> x >> y) {
        cities.push_back({ x, y });
    }
    return cities;
}

double calculate_distance(const City& city1, const City& city2) {
    return sqrt(pow(city1.x - city2.x, 2) + pow(city1.y - city2.y, 2));
}

double total_distance(const vector<int>& tour, const vector<City>& cities) {
    double distance = 0.0;
    for (size_t i = 1; i < tour.size(); ++i) {
        distance += calculate_distance(cities[tour[i]], cities[tour[i - 1]]);
    }
    return distance;
}

vector<int> generate_individual(int num_cities) {
    vector<int> individual(num_cities);
    std::iota(individual.begin(), individual.end(), 0);
    shuffle(individual.begin(), individual.end(), mt19937(random_device()()));
    return individual;
}

vector<vector<int>> generate_population(int population_size, int num_cities) {
    vector<vector<int>> population;
    for (int i = 0; i < population_size; ++i) {
        population.push_back(generate_individual(num_cities));
    }
    return population;
}

double fitness(const vector<int>& individual, const vector<City>& cities) {
    return 1.0 / total_distance(individual, cities);
}

pair<vector<int>, vector<int>> tournament_selection(const vector<vector<int>>& population, const vector<double>& fitnesses, int K) {
    std::random_device rd;
    std::mt19937 gen(rd());
    uniform_int_distribution<> dist(0, population_size - 1);

    int best_index1 = dist(gen), best_index2 = dist(gen);
    for (int i = 0; i < K - 1; ++i) {
        int idx = dist(gen);
        if (fitnesses[idx] > fitnesses[best_index1]) best_index1 = idx;
    }
    for (int i = 0; i < K - 1; ++i) {
        int idx = dist(gen);
        if (fitnesses[idx] > fitnesses[best_index2]) best_index2 = idx;
    }
    return { population[best_index1], population[best_index2] };
}

vector<int> order_crossover(const vector<int>& parent1, const vector<int>& parent2) {
    int size = parent1.size();
    std::random_device rd;
    std::mt19937 gen(rd());
    uniform_int_distribution<> dist(0, size - 1);
    int start = dist(gen), end = dist(gen);
    if (start > end) swap(start, end);

    vector<int> child(size, -1);
    for (int i = start; i <= end; ++i) {
        child[i] = parent1[i];
    }

    int cur = 0;
    for (int i = 0; i < size; ++i) {
        if (find(child.begin(), child.end(), parent2[i]) == child.end()) {
            while (cur >= start && cur <= end) cur++;
            if (cur >= size) break;
            child[cur++] = parent2[i];
        }
    }
    return child;
}

vector<int> mutate(vector<int> individual) {
    std::random_device rd;
    std::mt19937 gen(rd());
    uniform_int_distribution<> dist(0, individual.size() - 1);
    int idx1 = dist(gen), idx2 = dist(gen);
    swap(individual[idx1], individual[idx2]);
    return individual;
}

vector<vector<int>> generate_offspring(const vector<vector<int>>& population, const vector<double>& fitnesses, const vector<City>& cities) {
    vector<vector<int>> offspring;
    for (int i = 0; i < population_size; ++i) {
        auto [parent1, parent2] = tournament_selection(population, fitnesses, K);
        vector<int> child = order_crossover(parent1, parent2);
        if (rand() / double(RAND_MAX) < mutation_rate) {
            child = mutate(child);
        }
        offspring.push_back(child);
    }
    return offspring;
}

double tsp_ga(const string& filename, int generations) {
    vector<City> cities = load_city_coordinates(filename);
    int num_cities = cities.size();
    auto population = generate_population(population_size, num_cities);
    double best_fitness = 0.0;

    for (int generation = 0; generation < generations; ++generation) {
        vector<double> fitnesses(population_size);
        for (int i = 0; i < population_size; ++i) {
            fitnesses[i] = fitness(population[i], cities);
        }
        auto max_it = max_element(fitnesses.begin(), fitnesses.end());
        best_fitness = *max_it;

        population = generate_offspring(population, fitnesses, cities);
    }

    return best_fitness;
}

int main() {
    string filename = "C:/dev/RU/CPM_collective_migration/Assigment_5/d198.tsp";
    int generations = 1500;
    std::vector<double> fitness;
    for (size_t i = 0; i < 10; i++)
    {
        double best_fitness = tsp_ga(filename, generations);
        fitness.push_back(best_fitness);
        cout << "Cycle: " << i << " Best fitness: " << best_fitness << endl;
    }
    auto average = std::accumulate(fitness.begin(), fitness.end(), 0.0)/fitness.size();
    std::cout << average << "\n" << fitness.size();

    return 0;
}
