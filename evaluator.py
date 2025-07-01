class FunctionEvaluator:
    """
    Esta classe monitora as chamadas para a função objetivo, contando
    o número de avaliações e as operações de multiplicação/divisão.
    """

    def __init__(self, objective_function):
        self.objective_function = objective_function
        self.reset()

    def evaluate(self, x, y):
        """
        Calcula a função e incrementa os contadores.
        """
        self.evaluations += 1
        # Contagem de operações para a função w22:
        # f(x,y) = (x - y) * (-x * sin(sqrt|x|) - y * sin(sqrt|y|))
        # 1. -x * sin(...) -> 1 multiplicação
        # 2. -y * sin(...) -> 1 multiplicação
        # 3. (x-y) * (...) -> 1 multiplicação
        # Total de 3 multiplicações. Nenhuma divisão.
        self.multiplications += 3
        return self.objective_function(x, y)

    def reset(self):
        self.evaluations = 0
        self.multiplications = 0
        self.divisions = 0

    def get_stats(self):
        return {
            "evaluations": self.evaluations,
            "multiplications": self.multiplications,
            "divisions": self.divisions
        }
