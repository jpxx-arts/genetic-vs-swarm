import numpy as np


class GeneticAlgorithm:
    def __init__(self, func_evaluator, bounds, pop_size=50, generations=100,
                 crossover_rate=0.8, mutation_rate=0.1):
        self.func_evaluator = func_evaluator
        self.bounds = bounds
        self.pop_size = pop_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate

        self.population = []
        self.best_solution = None
        self.best_fitness = float('inf')
        self.convergence_info = None
        self.global_min_info = None

        self.internal_multiplications = 0
        self.internal_divisions = 0

    def _initialize_population(self):
        self.population = np.random.uniform(
            self.bounds[0], self.bounds[1], (self.pop_size, 2)
        ).tolist()

    def _evaluate_individual(self, individual):
        x, y = individual

        return self.func_evaluator.evaluate(x, y)

    def _selection(self, fitnesses):
        fitness_array = np.array(fitnesses)
        max_fitness = np.max(fitness_array)

        inverted_weights = (max_fitness - fitness_array) + 1

        total_weight = np.sum(inverted_weights)
        if total_weight == 0:
            probabilities = np.full(self.pop_size, 1 / self.pop_size)
        else:
            probabilities = inverted_weights / total_weight
            self.internal_divisions += self.pop_size

        selected_indices = np.random.choice(
            a=self.pop_size,
            size=self.pop_size,
            replace=True,
            p=probabilities
        )

        selected_population = [self.population[i] for i in selected_indices]
        return selected_population

    def _crossover(self, parent1, parent2):
        child1, child2 = parent1[:], parent2[:]

        if np.random.rand() < self.crossover_rate:
            # CONTANDO AS MULTIPLICAÇÕES NO CROSSOVER
            # Para child1: alpha * p1[0], (1-a)*p2[0], alpha*p1[1], (1-a)*p2[1] -> 4 mult.
            # Para child2: alpha * p2[0], (1-a)*p1[0], alpha*p2[1], (1-a)*p1[1] -> 4 mult.
            # Total: 8 multiplicações.
            self.internal_multiplications += 8

            alpha = np.random.rand()
            child1[0] = alpha * parent1[0] + (1 - alpha) * parent2[0]
            child1[1] = alpha * parent1[1] + (1 - alpha) * parent2[1]
            child2[0] = alpha * parent2[0] + (1 - alpha) * parent1[0]
            child2[1] = alpha * parent2[1] + (1 - alpha) * parent1[1]

        return child1, child2

    def _mutate(self, individual):
        for i in range(len(individual)):
            if np.random.rand() < self.mutation_rate:
                # Adiciona ruído gaussiano
                mutation_value = np.random.normal(0, 5)  # desvio padrão de 5
                individual[i] += mutation_value

                # Garante que o indivíduo permaneça dentro dos limites
                individual[i] = np.clip(
                    individual[i], self.bounds[0], self.bounds[1])

        return individual

    def run(self):
        self.internal_multiplications = 0
        self.internal_divisions = 0

        self._initialize_population()

        last_improvement_gen = 0

        for gen in range(self.generations):
            # 1. Avaliação
            fitnesses = [self._evaluate_individual(
                ind) for ind in self.population]

            # 2. Rastreamento do melhor resultado
            current_best_idx = np.argmin(fitnesses)

            if fitnesses[current_best_idx] < self.best_fitness:
                self.best_fitness = fitnesses[current_best_idx]
                self.best_solution = self.population[current_best_idx]
                self.global_min_info = (self.func_evaluator.get_stats(),
                                        self.internal_multiplications, self.internal_divisions)
                last_improvement_gen = gen

            # 3. Seleção
            selected_population = self._selection(fitnesses)

            # 4. Crossover e Mutação
            next_population = []
            for i in range(0, self.pop_size, 2):
                parent1, parent2 = selected_population[i], selected_population[i+1]
                child1, child2 = self._crossover(parent1, parent2)
                next_population.append(self._mutate(child1))
                next_population.append(self._mutate(child2))

            self.population = next_population

            # Critério de convergência: se não houver melhora por 20 gerações
            if gen - last_improvement_gen > 20 and self.convergence_info is None:
                self.convergence_info = (self.func_evaluator.get_stats(),
                                         self.internal_multiplications,
                                         self.internal_divisions)

        if self.convergence_info is None:
            self.convergence_info = (self.func_evaluator.get_stats(),
                                     self.internal_multiplications,
                                     self.internal_divisions)
