import numpy as np


class ParticleSwarmOptimization:
    def __init__(self, func_evaluator, bounds, swarm_size=50, iterations=100,
                 w=0.5, c1=1.5, c2=1.5):
        self.func_evaluator = func_evaluator
        self.bounds = bounds
        self.swarm_size = swarm_size
        self.iterations = iterations
        self.w = w    # Inércia
        self.c1 = c1  # Coeficiente Cognitivo
        self.c2 = c2  # Coeficiente Social

        self.particles_pos = None
        self.particles_vel = None
        self.particles_pbest_pos = None
        self.particles_pbest_val = np.full(self.swarm_size, float('inf'))
        self.gbest_pos = None
        self.gbest_val = float('inf')
        self.convergence_info = None
        self.global_min_info = None

        self.internal_multiplications = 0
        self.internal_divisions = 0

    def _initialize_swarm(self):
        self.particles_pos = np.random.uniform(
            self.bounds[0], self.bounds[1], (self.swarm_size, 2))
        self.particles_vel = np.random.uniform(-1, 1, (self.swarm_size, 2))
        self.particles_pbest_pos = self.particles_pos.copy()

    def run(self):
        self.internal_multiplications = 0
        self.internal_divisions = 0

        self._initialize_swarm()
        last_improvement_iter = 0

        for it in range(self.iterations):
            for i in range(self.swarm_size):
                current_val = self.func_evaluator.evaluate(
                    self.particles_pos[i, 0], self.particles_pos[i, 1])

                if current_val < self.particles_pbest_val[i]:
                    self.particles_pbest_val[i] = current_val
                    self.particles_pbest_pos[i] = self.particles_pos[i].copy()

                if current_val < self.gbest_val:
                    self.gbest_val = current_val
                    self.gbest_pos = self.particles_pos[i].copy()
                    self.global_min_info = (self.func_evaluator.get_stats(),
                                            self.internal_multiplications, self.internal_divisions)
                    last_improvement_iter = it

            for i in range(self.swarm_size):
                r1, r2 = np.random.rand(2), np.random.rand(2)

                # CONTANDO AS MULTIPLICAÇÕES NA ATUALIZAÇÃO DE VELOCIDADE
                # Para cada partícula (vetor de 2 dimensões):
                # w * vel -> 2 mult.
                # c1 * r1 -> 2 mult.
                # (c1*r1) * (pbest - pos) -> 2 mult.
                # c2 * r2 -> 2 mult.
                # (c2*r2) * (gbest - pos) -> 2 mult.
                # Total: 10 multiplicações por partícula.
                self.internal_multiplications += 10

                cognitive_vel = self.c1 * r1 * \
                    (self.particles_pbest_pos[i] - self.particles_pos[i])
                social_vel = self.c2 * r2 * \
                    (self.gbest_pos - self.particles_pos[i])
                self.particles_vel[i] = self.w * \
                    self.particles_vel[i] + cognitive_vel + social_vel

                self.particles_pos[i] += self.particles_vel[i]
                self.particles_pos[i] = np.clip(
                    self.particles_pos[i], self.bounds[0], self.bounds[1])

            if it - last_improvement_iter > 20 and self.convergence_info is None:
                self.convergence_info = (self.func_evaluator.get_stats(),
                                         self.internal_multiplications, self.internal_divisions)

        if self.convergence_info is None:
            self.convergence_info = (self.func_evaluator.get_stats(),
                                     self.internal_multiplications, self.internal_divisions)
