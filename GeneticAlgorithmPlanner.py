import random
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from copy import deepcopy
class GANPlanner:
    def __init__(self, sleep_quality_path, ids_path, nb_children=4, nb_generation=200, max=False):
        '''
            Setting up the model
        :param sleep_quality_path: str
            The path to the binary file with the sleep quality predictor.
        :param ids_path: str
            The path to the binary file with the IDS predictor.
        :param nb_children: integer, default = 4
            The number of children created at every generation
        :param nb_generation: integer, default = 200
            The number of generations created to find the best feature combination
        :param max: boolean, default = True
            Used to define whatever we need the biggest of the smallest values computed by self.scoring_metric
        '''
        self.sleep_quality_scorer = pickle.load(open(sleep_quality_path, 'rb'))
        self.ids_scorer = pickle.load(open(ids_path, 'rb'))
        self.nb_children = nb_children
        self.nb_generation = nb_generation
        self.max = max
    def inverse(self, value):
        '''
            This function inverses the value coming to function
        :param value: integer, o or 1
            The value that should be inverted
        :return: integer
            Return the inverse of :param value (if value = 0 ==> 1, else 0)
        '''
        if value == 0:
            return 1
        else:
            return 0
    def cross_over(self, population):
        '''
            This function apply the crossing-over process on 2 arrys-like lists
        :param population: 2-d array like list
            The population with 2 array that will be uses to create new individuals in the population
        :return: 2-d array like list
            Return the population with parents and their children after corssing-over
        '''
        new_generation = []
        for i in range(self.nb_children//2):
            first = random.randrange(0, len(population[0])-1)
            second = random.randrange(0, len(population[0])-1)
            if first > second:
                first, second = second, first
            new_generation.append(population[0][0:first] + population[1][first: second] + population[0][second:])
            new_generation.append(population[1][0:first] + population[0][first: second] + population[1][second:])
        for gene in new_generation:
            population.append(gene)
        return population
    def mutate(self, gene):
        '''
            This function generates a random mutation on a gene
        :param gene: 1-d array like list
            The list with zeros and ones that will be mutated
        :return: 1-drray like list
            The gene list after mutation
        '''
        mutation_locus = random.randrange(0, len(gene)-1)
        gene[mutation_locus] = self.inverse(gene[mutation_locus])
        return gene


    def fit_function(self, slept_hours, imc, day_schedule, generated_schedule):
        '''
            The fitness function - the metric that defines how well schedule generator works.
        :param slept_hours: int
            The number of hours tha the person will sleep.
        :param imc: float
            The body mass index.
        :param day_schedule: list
            The day schedule of the person.
        :param generated_schedule: list
            The generated schedule
        :return:
        '''
        norm = np.linalg.norm(np.array(generated_schedule) - np.array(day_schedule), ord=2)
        ids = self.ids_scorer.predict([[slept_hours*60, imc]])[0]
        sleep_quality = self.sleep_quality_scorer.predict([[slept_hours]])[0]
        for i in range(1, len(generated_schedule)-1):
            if generated_schedule[i] > generated_schedule[i+1] and generated_schedule[i] > generated_schedule[i-1]:
                norm *= norm*20
            if generated_schedule[i] < generated_schedule[i+1] and generated_schedule[i] < generated_schedule[i-1]:
                norm *= norm*20
            if generated_schedule[0] != generated_schedule[-1]:
                norm *= norm*20
        if sum(generated_schedule[6:22]) > 2:
            norm += 1000
        if sum(generated_schedule[:6] + generated_schedule[-2:]) < len(generated_schedule[:6] + generated_schedule[-2:]):
            norm += 500
        return 30*norm * ids/(sleep_quality * 100)

    def generate(self, day_schedule, imc, warm_start = True):
        '''
            This function generates the sleep regime
        :param day_schedule: list
            The day schedule of the person.
        :param imc: float
            The body mass index.
        :param warm_start: bool, default = True
            Defines whether
        :return: list
            The generated schedule
        '''
        if warm_start == 1:
            sleep_schedule = [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
        else:
            sleep_schedule = list(np.random.choice([0, 1], size=24, p=[.5, .5]))
        population = [sleep_schedule]
        for i in range(3):
            population.append(self.mutate(population[i]))
        for gen in range(self.nb_generation):
            fit_quality = []
            population = self.cross_over(population)
            for i in range(2, len(population)):
                population[i] = self.mutate(population[i])
            for genotype in population:
                print(genotype)
                fit_quality.append(self.fit_function(sum(genotype), imc, day_schedule, genotype))
            print(sorted(fit_quality))
            if self.max:
                best = sorted(range(len(fit_quality)), key=lambda sub: fit_quality[sub])[-2:]
            else:
                best = sorted(range(len(fit_quality)), key=lambda sub: fit_quality[sub])[:2]
            population = [population[best[0]], population[best[1]]]
        return population[0]