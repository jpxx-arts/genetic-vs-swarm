import math
import time
import genetic_alg as ga
import particle_swarm as ps
import evaluator as eval


def w22(x, y):
    z = -x * math.sin(math.sqrt(abs(x))) + -y * math.sin(math.sqrt(abs(y)))
    return (x - y) * z


if __name__ == "__main__":
    BOUNDS = [-500, 500]
    evaluator = eval.FunctionEvaluator(objective_function=w22)

    print("--- Iniciando Algoritmo Genético ---")
    evaluator.reset()
    ga = ga.GeneticAlgorithm(
        func_evaluator=evaluator, bounds=BOUNDS, pop_size=100, generations=200,
        crossover_rate=0.8, mutation_rate=0.2
    )
    start_time = time.time()
    ga.run()
    ga_time = time.time() - start_time

    print("\nResultados do Algoritmo Genético")
    print(f"Tempo de execução: {ga_time:.4f} segundos")
    print(f"Melhor solução (x, y): ({
          ga.best_solution[0]:.4f}, {ga.best_solution[1]:.4f})")
    print(f"Valor mínimo da função: {ga.best_fitness:.4f}")

    ga_conv_stats, ga_conv_mult_int, ga_conv_div_int = ga.convergence_info
    ga_min_stats, ga_min_mult_int, ga_min_div_int = ga.global_min_info

    print("\nAnálise Computacional")
    print("a) Convergência alcançada com:")
    print(f"  - Avaliações da Função: {ga_conv_stats['evaluations']}")
    print(f"  - Operações na Função: {ga_conv_stats['multiplications']} mult, {
          ga_conv_stats['divisions']} div")
    print(
        f"  - Operações Internas GA: {ga_conv_mult_int} mult, {ga_conv_div_int} div")
    print(f"  - TOTAIS: {ga_conv_stats['multiplications'] + ga_conv_mult_int} mult, {
          ga_conv_stats['divisions'] + ga_conv_div_int} div")

    print("\nb) Mínimo global encontrado com:")
    print(f"  - Avaliações da Função: {ga_min_stats['evaluations']}")
    print(f"  - Operações na Função: {ga_min_stats['multiplications']} mult, {
          ga_min_stats['divisions']} div")
    print(
        f"  - Operações Internas GA: {ga_min_mult_int} mult, {ga_min_div_int} div")
    print(f"  - TOTAIS: {ga_min_stats['multiplications'] + ga_min_mult_int} mult, {
          ga_min_stats['divisions'] + ga_min_div_int} div")

    print("\n\n--- Iniciando Otimização por Enxame de Partículas ---")
    evaluator.reset()
    pso = ps.ParticleSwarmOptimization(
        func_evaluator=evaluator, bounds=BOUNDS, swarm_size=100, iterations=200,
        w=0.5, c1=2.0, c2=2.0
    )
    start_time = time.time()
    pso.run()
    pso_time = time.time() - start_time

    print("\nResultados da Otimização por Enxame de Partículas")
    print(f"Tempo de execução: {pso_time:.4f} segundos")
    print(f"Melhor solução (x, y): ({
          pso.gbest_pos[0]:.4f}, {pso.gbest_pos[1]:.4f})")
    print(f"Valor mínimo da função: {pso.gbest_val:.4f}")

    pso_conv_stats, pso_conv_mult_int, pso_conv_div_int = pso.convergence_info
    pso_min_stats, pso_min_mult_int, pso_min_div_int = pso.global_min_info

    print("\nAnálise Computacional")
    print("a) Convergência alcançada com:")
    print(f"  - Avaliações da Função: {pso_conv_stats['evaluations']}")
    print(f"  - Operações na Função: {pso_conv_stats['multiplications']} mult, {
          pso_conv_stats['divisions']} div")
    print(
        f"  - Operações Internas PSO: {pso_conv_mult_int} mult, {pso_conv_div_int} div")
    print(f"  - TOTAIS: {pso_conv_stats['multiplications'] + pso_conv_mult_int} mult, {
          pso_conv_stats['divisions'] + pso_conv_div_int} div")

    print("\nb) Mínimo global encontrado com:")
    print(f"  - Avaliações da Função: {pso_min_stats['evaluations']}")
    print(f"  - Operações na Função: {pso_min_stats['multiplications']} mult, {
          pso_min_stats['divisions']} div")
    print(
        f"  - Operações Internas PSO: {pso_min_mult_int} mult, {pso_min_div_int} div")
    print(f"  - TOTAIS: {pso_min_stats['multiplications'] + pso_min_mult_int} mult, {
          pso_min_stats['divisions'] + pso_min_div_int} div")

    print("\n\n--- Comparação Final de Desempenho ---\n")
    ga_total_ops = ga_min_stats['multiplications'] + \
        ga_min_mult_int + ga_min_stats['divisions'] + ga_min_div_int
    pso_total_ops = pso_min_stats['multiplications'] + \
        pso_min_mult_int + pso_min_stats['divisions'] + pso_min_div_int

    print(f"Operações totais para encontrar o mínimo (Algoritmo Genético): {ga_total_ops}")
    print(f"Operações totais para encontrar o mínimo (Enxame de Partículas): {pso_total_ops}")

    if pso.gbest_val < ga.best_fitness:
        print("\nO PSO encontrou um valor mínimo melhor.\n")
    else:
        print("\nO GA encontrou um valor mínimo melhor.\n")

    if pso_total_ops < ga_total_ops:
        print("Considerando o número total de operações, o PSO apresentou melhor desempenho computacional.")
    else:
        print("Considerando o número total de operações, o GA apresentou melhor desempenho computacional.")
