import csv
import time
import os
import numpy as np

import evaluator as ev
import genetic_alg as ga
import particle_swarm as ps


def w22(x, y):
    term1 = -x * np.sin(np.sqrt(np.abs(x)))
    term2 = -y * np.sin(np.sqrt(np.abs(y)))
    return (x - y) * (term1 + term2)


ga_param_space = [
    # Estratégia: População massiva para máxima diversidade inicial.
    {'pop_size': 400, 'generations': 75, 'crossover_rate': 0.8, 'mutation_rate': 0.3},

    # Estratégia: Mutação muito alta para forçar a saída de mínimos locais.
    {'pop_size': 150, 'generations': 200,
        'crossover_rate': 0.85, 'mutation_rate': 0.5},

    # Estratégia: População grande e mutação alta, um equilíbrio exploratório.
    {'pop_size': 300, 'generations': 100,
        'crossover_rate': 0.9, 'mutation_rate': 0.4},

    # Estratégia: "Slow burn" - Crossover mais baixo para preservar boas mutações, com mais tempo para evoluir.
    {'pop_size': 200, 'generations': 150,
        'crossover_rate': 0.6, 'mutation_rate': 0.2},

    # Estratégia: Taxa de mutação extremamente alta, quase uma busca aleatória guiada.
    {'pop_size': 200, 'generations': 150,
        'crossover_rate': 0.7, 'mutation_rate': 0.7},

    # Estratégia: Variação da população massiva com alta taxa de crossover.
    {'pop_size': 500, 'generations': 60, 'crossover_rate': 0.9, 'mutation_rate': 0.3},

    # Estratégia: Balanceado, mas com mais gerações e mutação moderada.
    {'pop_size': 150, 'generations': 200,
        'crossover_rate': 0.8, 'mutation_rate': 0.25},

    # Estratégia: População massiva para máxima diversidade inicial.
    {'pop_size': 400, 'generations': 75, 'crossover_rate': 0.8, 'mutation_rate': 0.3},

    # Estratégia: Mutação muito alta para forçar a saída de mínimos locais.
    {'pop_size': 150, 'generations': 200,
        'crossover_rate': 0.85, 'mutation_rate': 0.5},

    # Estratégia: População grande e mutação alta, um equilíbrio exploratório.
    {'pop_size': 300, 'generations': 100,
        'crossover_rate': 0.9, 'mutation_rate': 0.4},

    # Estratégia: "Slow burn" - Crossover mais baixo para preservar boas mutações, com mais tempo para evoluir.
    {'pop_size': 200, 'generations': 150,
        'crossover_rate': 0.6, 'mutation_rate': 0.2},

    # Estratégia: Taxa de mutação extremamente alta, quase uma busca aleatória guiada.
    {'pop_size': 200, 'generations': 150,
        'crossover_rate': 0.7, 'mutation_rate': 0.7},

    # Estratégia: Variação da população massiva com alta taxa de crossover.
    {'pop_size': 500, 'generations': 60, 'crossover_rate': 0.9, 'mutation_rate': 0.3},

    # Estratégia: Balanceado, mas com mais gerações e mutação moderada.
    {'pop_size': 150, 'generations': 200,
        'crossover_rate': 0.8, 'mutation_rate': 0.25},
]

pso_param_space = [
    {'swarm_size': 15, 'iterations': 20, 'w': 0.001, 'c1': 2.0, 'c2': 2.0}, # Inércia quase zero
    {'swarm_size': 15, 'iterations': 20, 'w': 0.1,   'c1': 2.0, 'c2': 2.0}, # Inércia muito baixa
    {'swarm_size': 15, 'iterations': 20, 'w': 0.3,   'c1': 2.0, 'c2': 2.0}, # Inércia baixa
    {'swarm_size': 15, 'iterations': 20, 'w': 0.5,   'c1': 2.0, 'c2': 2.0}, # Inércia moderada
    {'swarm_size': 15, 'iterations': 20, 'w': 0.7,   'c1': 2.0, 'c2': 2.0}, # Inércia alta (foco em exploração)

    {'swarm_size': 40, 'iterations': 10, 'w': 0.5, 'c1': 2.0, 'c2': 2.0}, # Busca larga e curta
    {'swarm_size': 20, 'iterations': 20, 'w': 0.5, 'c1': 2.0, 'c2': 2.0}, # Busca balanceada
    {'swarm_size': 10, 'iterations': 40, 'w': 0.5, 'c1': 2.0, 'c2': 2.0}, # Busca estreita e longa
    {'swarm_size': 8,  'iterations': 50, 'w': 0.5, 'c1': 2.0, 'c2': 2.0}, # Busca muito estreita e muito longa

    {'swarm_size': 12, 'iterations': 30, 'w': 0.4, 'c1': 2.5, 'c2': 1.5}, # Mais individualista
    {'swarm_size': 12, 'iterations': 30, 'w': 0.4, 'c1': 1.5, 'c2': 2.5}, # Mais social (convergência rápida)
]

# --- EXECUÇÃO E COLETA DE DADOS ---
results_filename = 'tuning_results.csv'

# Cabeçalho do arquivo CSV corrigido e unificado
csv_header = [
    'algorithm', 'pop_or_swarm_size', 'gens_or_iterations', 'crossover_rate', 'mutation_rate',
    'w', 'c1', 'c2', 'best_fitness', 'evaluations_to_find_min', 'total_ops_to_find_min', 'execution_time'
]

# Verifica se o arquivo já existe para não reescrever o cabeçalho
file_exists = os.path.isfile(results_filename)

with open(results_filename, mode='a', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=csv_header)
    if not file_exists:
        writer.writeheader()

    evaluator = ev.FunctionEvaluator(objective_function=w22)
    BOUNDS = [-500, 500]

    # --- Loop de Teste para o Algoritmo Genético ---
    print("--- INICIANDO TESTES DO ALGORITMO GENÉTICO ---")
    for i, params in enumerate(ga_param_space):
        print(
            f"Executando GA - Teste {i+1}/{len(ga_param_space)} com parâmetros: {params}")

        evaluator.reset()
        ga_instance = ga.GeneticAlgorithm(
            func_evaluator=evaluator, bounds=BOUNDS, **params)

        start_time = time.time()
        ga_instance.run()
        end_time = time.time()

        # Coleta de resultados
        stats_min, mult_int, div_int = ga_instance.global_min_info

        writer.writerow({
            'algorithm': 'GA', 'pop_or_swarm_size': params['pop_size'], 'gens_or_iterations': params['generations'],
            'crossover_rate': params['crossover_rate'], 'mutation_rate': params['mutation_rate'],
            'best_fitness': ga_instance.best_fitness,
            'evaluations_to_find_min': stats_min['evaluations'],
            'total_ops_to_find_min': stats_min['multiplications'] + mult_int + stats_min['divisions'] + div_int,
            'execution_time': end_time - start_time
        })

    # --- Loop de Teste para o Enxame de Partículas ---
    print("\n--- INICIANDO TESTES DO ENXAME DE PARTÍCULAS ---")
    for i, params in enumerate(pso_param_space):
        print(
            f"Executando PSO - Teste {i+1}/{len(pso_param_space)} com parâmetros: {params}")

        evaluator.reset()
        pso_instance = ps.ParticleSwarmOptimization(
            func_evaluator=evaluator, bounds=BOUNDS, **params)

        start_time = time.time()
        pso_instance.run()
        end_time = time.time()

        # Coleta de resultados
        stats_min, mult_int, div_int = pso_instance.global_min_info

        writer.writerow({
            'algorithm': 'PSO', 'pop_or_swarm_size': params['swarm_size'], 'gens_or_iterations': params['iterations'],
            'w': params['w'], 'c1': params['c1'], 'c2': params['c2'],
            'best_fitness': pso_instance.gbest_val,
            'evaluations_to_find_min': stats_min['evaluations'],
            'total_ops_to_find_min': stats_min['multiplications'] + mult_int + stats_min['divisions'] + div_int,
            'execution_time': end_time - start_time
        })

print("\n--- TESTES CONCLUÍDOS. Resultados salvos em 'tuning_results.csv' ---")

# --- ANÁLISE FINAL DOS RESULTADOS ---
print("\n--- ANÁLISE DOS MELHORES RESULTADOS ---")
best_ga_run = None
best_pso_run = None

with open(results_filename, mode='r', newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        # Converte valores para numérico para comparação
        row['best_fitness'] = float(row['best_fitness'])

        if row['algorithm'] == 'GA':
            if best_ga_run is None or row['best_fitness'] < best_ga_run['best_fitness']:
                best_ga_run = row
        elif row['algorithm'] == 'PSO':
            if best_pso_run is None or row['best_fitness'] < best_pso_run['best_fitness']:
                best_pso_run = row

if best_ga_run:
    print("\nMelhor Configuração encontrada para o Algoritmo Genético:")
    print(f"  - Fitness: {best_ga_run['best_fitness']:.4f}")
    print(f"  - Parâmetros: pop_size={best_ga_run['pop_or_swarm_size']}, generations={
          best_ga_run['gens_or_iterations']}, crossover_rate={best_ga_run['crossover_rate']}, mutation_rate={best_ga_run['mutation_rate']}")
    print(f"  - Custo: {best_ga_run['total_ops_to_find_min']
                        } operações em {float(best_ga_run['execution_time']):.2f}s")

if best_pso_run:
    print("\nMelhor Configuração encontrada para o Enxame de Partículas:")
    print(f"  - Fitness: {best_pso_run['best_fitness']:.4f}")
    print(f"  - Parâmetros: swarm_size={best_pso_run['pop_or_swarm_size']}, iterations={
          best_pso_run['gens_or_iterations']}, w={best_pso_run['w']}, c1={best_pso_run['c1']}, c2={best_pso_run['c2']}")
    print(f"  - Custo: {best_pso_run['total_ops_to_find_min']
                        } operações em {float(best_pso_run['execution_time']):.2f}s")
