# =====================================================================================================
# GA_Optimization_v2.py -- GA_Optimization.py with the issues found in the revision fixed (3 October 2026).
# Every change is marked "# CHANGE n". The search itself (operators, eaSimple, parameters) is unchanged.
#
#   CHANGE 1  Evaluations go through SimulationInterface_fast: feasibility checked before the cache lookup, distributions
#             up to 2000 with an infinite cost if one is cut off, the compiled simulator. (The original went through
#             the same interface as the PSO, with the same two defects.)
#   CHANGE 2  The reported cost of the best design comes from an INDEPENDENT evaluation (35 replications, seeds
#             final_seed + 20 i, no cache), not from the seeds that selected it; utilization and assignment statistics
#             are the design's own. The search cost is kept as log.search_total_cost.
#   CHANGE 3  np.random.random_integers (deprecated since NumPy 1.11) -> np.random.randint(low, high + 1): the same
#             draws, so seeded runs are unchanged.
#   CHANGE 4  The "avg" and "maxninf" statistics no longer fail when every individual of a generation is infeasible.
# Already correct and unchanged: the seed is used (random and np.random); the best design is tracked over all
# generations (HallOfFame), including the last; one individual of the initial population is full cross-training.
# =====================================================================================================
#    This file is part of DEAP.
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.

import random
import sys

from simulation_codes.SimulationInterface_fast import simulation_optimization_run, initDatabase   # CHANGE 1
from simulation_codes import SimulationInterface_fast as SIf
TRACK = {"cost": float("inf"), "X": None}       # best design evaluated (used if the budget stops the search)
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
import numpy as np
import time
import json

# from EHBapproximation.EHBapproximation import EHBapproximation


# @profile
def evaluateIndividualCost(individual, failure_rates, service_rates, holding_costs, penalty_cost, machine_cost, skill_cost, runTheSimulation=True):
    '''
    :param individual: the server assignment of the tested individual
    :param n_servers: Total number of machines/servers
    :param failure_rate: Arrival rate of each SKU to repair-shop (list of size n_skill)
    :param service_rate: Service rate of each SKU at any machine (list of size n_skill-service rate are independent of machine)
    :param holding_cost: Inventory holding cost SKU per time unit per part (list of size n_skill)
    :param penalty_cost: Penalty cost for each back ordered demand per time unit (assumed to be same for every SKU)
    :param machine_cost: Operation cost for a machine (list of size m_machine)
    :param skill_cost: Cost of having skill to repair SKUs per unit time (list of size n_skill)
    :return: Total cost and cost terms of the curent individual

    the costs are evaluated based on the simulation/database combination

    '''

    m_skills = len(failure_rates)

    totalCost = float("inf")
    assignment = np.array(individual).reshape(-1, m_skills)  # converts into matrix form if assignment is a list
    totalCost, stock, ebo, databaseUpdated, _, _ = simulation_optimization_run(failure_rates, service_rates, holding_costs, penalty_cost, assignment, runTheSimulation=runTheSimulation, stopNrRequests=100000, replications=35)
    totalCost += skill_cost*np.sum(assignment)
    totalCost +=  machine_cost*sum((np.sum(assignment,axis=1)>0))
    if totalCost < TRACK["cost"]:
        TRACK["cost"], TRACK["X"] = totalCost, assignment.copy()
        TRACK.setdefault("trace", []).append((SIf.simulated, float(totalCost)))

    # print "databaseUpdated:", databaseUpdated, "| assignment:", assignment.tolist(), "| TotalCost", totalCost

    stock_cost = sum(stock * holding_costs)
    ebo_cost = sum(ebo*penalty_cost)
    servers_cost = machine_cost*sum((np.sum(assignment,axis=1)>0))
    skills_cost = skill_cost*np.sum(assignment)


    return totalCost, stock_cost, ebo_cost, servers_cost, skills_cost # None, databaseUpdated

def evaluateIndividualCostApprox(individual, failure_rates, service_rates, holding_costs, penalty_cost, machine_cost, skill_cost, runTheSimulation=False):
    '''
    :param individual: the server assignment of the tested individual
    :param n_servers: Total number of machines/servers
    :param failure_rate: Arrival rate of each SKU to repair-shop (list of size n_skill)
    :param service_rate: Service rate of each SKU at any machine (list of size n_skill-service rate are independent of machine)
    :param holding_cost: Inventory holding cost SKU per time unit per part (list of size n_skill)
    :param penalty_cost: Penalty cost for each back ordered demand per time unit (assumed to be same for every SKU)
    :param machine_cost: Operation cost for a machine (list of size m_machine)
    :param skill_cost: Cost of having skill to repair SKUs per unit time (list of size n_skill)
    :return: Total cost and cost terms of the curent individual

    the costs are evaluated based on the simulation/database combination

    '''

    m_skills = len(failure_rates)

    totalCost = float("inf")
    assignment = np.array(individual).reshape(-1, m_skills)  # converts into matrix form if assignment is a list
    totalCost, stock, ebo = EHBapproximation(failure_rates, service_rates, holding_costs, penalty_cost, assignment)
    totalCost += skill_cost*np.sum(assignment)
    totalCost +=  machine_cost*sum((np.sum(assignment,axis=1)>0))

    # print "databaseUpdated:", databaseUpdated, "| assignment:", assignment.tolist(), "| TotalCost", totalCost

    stock_cost = sum(stock * holding_costs)
    ebo_cost = sum(ebo*penalty_cost)
    servers_cost = machine_cost*sum((np.sum(assignment,axis=1)>0))
    skills_cost = skill_cost*np.sum(assignment)


    return totalCost, stock_cost, ebo_cost, servers_cost, skills_cost # None, databaseUpdated

def createRandomAssignment(lambdas, mus, n_servers):
    assignment = []
    n = np.ceil(np.array(lambdas)/mus)
    m_skills = len(lambdas)

    for i in range(m_skills):
        n_i = np.random.randint(int(n[i]), n_servers + 1)                  # CHANGE 3 (same draws)
        assignment.append(np.random.permutation(np.array([1]*n_i + [0]*(n_servers-n_i))).tolist())

    assignment = map(list, zip(*assignment)) #transpose the obtained list
    # print assignment

    return assignment # sum(assignment, []) # the sum flattens the list of lists

def mutateFlipBit2D(individual, indpb):
    for x in individual:
        for i in range(len(x)):
            if random.random() < indpb:
                x[i] = type(x[i])(not x[i])

    return individual,


def crossOverTwoPoints_servers(ind1, ind2):

    size = len(ind1)  # assume both individuals are of the same size
    cxpoint1 = random.randint(1, size)
    cxpoint2 = random.randint(1, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else: # Swap the two cx points
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] \
        = ind2[cxpoint1:cxpoint2], ind1[cxpoint1:cxpoint2]

    return ind1, ind2

def crossOverTwoPoints_skills(ind1, ind2):

    size = len(ind1[0])  # assume both individuals are of the same size
    cxpoint1 = random.randint(1, size)
    cxpoint2 = random.randint(1, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else: # Swap the two cx points
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1


    for i in range(len(ind1)):
        ind1[i][cxpoint1:cxpoint2], ind2[i][cxpoint1:cxpoint2] \
            = ind2[i][cxpoint1:cxpoint2], ind1[i][cxpoint1:cxpoint2]

    return ind1, ind2



# @profile
def optimizeSkillAssignment(n_servers, failure_rates, service_rates, holding_costs, penalty_cost, machine_cost, skill_cost,
                            ga_params = {'population': 100, 'generation': 50, 'crossover': 0.4, 'mutation': 0.1, 'indpb': 0.1, 'tournsize': 10},
                            #population = 100, generation = 50, crossover = 0.4, mutation = 0.1, indpb=0.1, tournsize=10,
                            seed = None, verbose = False, runTheSimulation=True, final_seed = 1_000_000):
    '''
    Experiment parameters
    :param n_servers: Total number of machines/servers
    :param failure_rate: Arrival rate of each SKU to repair-shop (list of size n_skill)
    :param service_rate: Service rate of each SKU at any machine (list of size n_skill-service rate are independent of machine)
    :param holding_cost: Inventory holding cost SKU per time unit per part (list of size n_skill)
    :param penalty_cost: Penalty cost for each back ordered demand per time unit (assumed to be same for every SKU)
    :param machine_cost: Operation cost for a machine (list of size m_machine)
    :param skill_cost: Cost of having skill to repair SKUs per unit time (list of size n_skill)
    GA related parameters
    :param population = 100: Population size in the GA procedure
    :param generation = 50: Number of generations in the GA procedure
    :param crossover = 0.4: Crossover probability, i.e. probability to mate two individuals (cxpb in eaSimple)
    :param mutation = 0.1: Mutation probability, i.e. probability with which an individual will be mutated (mutpb in eaSimple)
    :param indpb = 0.1: Probability to flip a bit if individual is selected for mutaiton (indpb in mutateFlipBit2D)
    :param tournsize = 10: Turnament size in selecting best individuals, if equal to population, then all selected will be equal to the best one
    :param seed = None:  Fixes seed in the random generators, can help in order to reproduce the same results in the GA optimization
    :param verbose = False: switches off output of the eaSimpel procedure
    :return: best_ind, totalCost, S, EBO, log

    the best individual is selected using eaSimple procedure in the DEAP library
    '''

    random.seed(seed)
    np.random.seed(seed)
    np.set_printoptions(linewidth=50000)

    creator.create("FitnessMax", base.Fitness, weights=(-1.0, -1.0, -1.0, -1.0, -1.0)) #-1 for minimization
    creator.create("Individual", list, fitness=creator.FitnessMax)


    toolbox = base.Toolbox()

    toolbox.register("assignment", createRandomAssignment, failure_rates, service_rates, n_servers)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.assignment)
    # toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, GA_n_server*GA_n_skills)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)


    # Operator registering
    # toolbox.register("evaluate", evaluateIndividualCost,
    toolbox.register("evaluate", evaluateIndividualCost, #Approx,
                                 # n_servers = n_servers,
                                 failure_rates = failure_rates,
                                 service_rates = service_rates,
                                 holding_costs = holding_costs,
                                 penalty_cost = penalty_cost,
                                 machine_cost = machine_cost,
                                 skill_cost = skill_cost,
                                 runTheSimulation = runTheSimulation)

    toolbox.register("mate", crossOverTwoPoints_skills)
    toolbox.register("mutate", mutateFlipBit2D, indpb=ga_params['indpb'])
    toolbox.register("select", tools.selTournament, tournsize=ga_params['tournsize'])

    pop = toolbox.population(n=ga_params['population'])

    rnd_ind = np.random.randint(0, len(pop))                               # CHANGE 3 (same draws)

    pop[rnd_ind][:] = np.ones((n_servers, len(failure_rates)), dtype=int).tolist() #overwrite a random individuum with the the fullflex assignement

    hof = tools.HallOfFame(1)
    tc_stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stock_stats = tools.Statistics(lambda ind: ind.fitness.values[1])
    ebo_stats = tools.Statistics(lambda ind: ind.fitness.values[2])
    server_stats = tools.Statistics(lambda ind: ind.fitness.values[3])
    skills_stats = tools.Statistics(lambda ind: ind.fitness.values[4])

    mstats = tools.MultiStatistics(total=tc_stats, stock_cost=stock_stats, ebo_cost=ebo_stats, server_cost=server_stats, skill_cost=skills_stats)

    mstats.register("avg", lambda x: np.mean(np.array(x)[~np.isinf(x)]) if not np.all(np.isinf(x)) else -1)   # CHANGE 4
    # stats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("maxninf", lambda x: max(np.array(x)[~np.isinf(x)]) if not np.all(np.isinf(x)) else -1)   # CHANGE 4
    mstats.register("max", np.max)
    mstats.register("median", np.median)
    mstats.register("Q25", np.percentile, q=25)
    mstats.register("Q75", np.percentile, q=75)

    TRACK["cost"], TRACK["X"], TRACK["trace"] = float("inf"), None, []
    try:
        pop, log = algorithms.eaSimple(pop, toolbox, cxpb=ga_params['crossover'], mutpb=ga_params['mutation'], ngen=ga_params['generation'],
                                       stats=mstats, halloffame=hof, verbose=verbose)
        best_ind = hof[0]
        del pop
        best_assignment = np.array(best_ind).reshape(-1, len(failure_rates))
        log.search_total_cost = best_ind.fitness.values[0]                  # CHANGE 2: cost with the search's seeds
        log.budget_stopped = False
    except SIf.BudgetSpent:                                                  # budget spent: best design evaluated
        log = tools.Logbook(); log.budget_stopped = True
        best_assignment = np.array(TRACK["X"]); log.search_total_cost = TRACK["cost"]

    totalCost, S, EBO, _, utilizationRatesAverage, skillServerAssignmentStatisitcsAverage = \
        simulation_optimization_run(failure_rates, service_rates, holding_costs, penalty_cost, best_assignment,
                                    runTheSimulation=runTheSimulation, useDatabase=False, startSeed=final_seed)   # CHANGE 2
    totalCost += skill_cost*np.sum(best_assignment)
    totalCost +=  machine_cost*sum((np.sum(best_assignment,axis=1)>0))


    del creator.FitnessMax
    del creator.Individual

    return best_assignment, totalCost, S, EBO, log, utilizationRatesAverage, skillServerAssignmentStatisitcsAverage



def experiments():
    # # c = csv.writer(open("MYFILE5.csv", "wb"))
    #
    # # file_path = '/Users/hasanturan/Dropbox/skills optimization (2)/result.json'
    # file_path = "result 2.json"
    #
    # with open(file_path, 'r') as data_file:
    #         data = json.load(data_file)
    #
    #
    # population_settings = [50, 100, 150, 200]
    # generation_settings = [5, 10, 20, 30]
    # mutation_settings = [0.1, 0.2, 0.4, 0.6]
    # crossover_settings = [0.2, 0.4, 0.6, 0.8, 0.9]
    #
    #
    # for (population, generation, crossover, mutation) in itertools.product(population_settings, generation_settings,crossover_settings, mutation_settings):
    #     maxGAP = 0.0
    #     for key, GA_iter in data.iteritems(): # .values():#range(len(data)):

    pass

def test1():

    initDatabase("db_simulations_test")

    n_servers = 2
    m_skills = 4
    failureRates = np.array([0.21415405370607182, 0.6858459462939283, 0.21415405370607182, 0.6858459462939283])
    serviceRates = np.array([1.0, 1.0, 1.0, 1.0])
    # failureRates = np.array([6]*m_skills, dtype=float)
    # serviceRates = np.array([13]*m_skills, dtype=float)
    holdingCosts = np.ones(m_skills)*1
    penalty = 100.0

    machineCost = 1000
    skillCost = 100

    ga_params = {'population': 10, 'generation': 10, 'crossover': 0.4, 'mutation': 0.1, 'indpb': 0.1, 'tournsize': 10}
    print(optimizeSkillAssignment(n_servers, failureRates, serviceRates, holdingCosts, penalty, machineCost, skillCost, ga_params=ga_params, seed=123, verbose=True))


def test2():

    initDatabase()

    n_servers = 3
    m_skills = 6
    failureRates = np.array([0.21415405370607182, 0.6858459462939283, 0.21415405370607182, 0.6858459462939283, 0.21415405370607182, 0.6858459462939283])
    serviceRates = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    # failureRates = np.array([6]*m_skills, dtype=float)
    # serviceRates = np.array([13]*m_skills, dtype=float)
    holdingCosts = np.ones(m_skills)*1
    penalty = 100.0

    machineCost = 1000
    skillCost = 100

    ga_params = {'population': 100, 'generation': 50, 'crossover': 0.4, 'mutation': 0.1, 'indpb': 0.1, 'tournsize': 10}
    print(optimizeSkillAssignment(n_servers, failureRates, serviceRates, holdingCosts, penalty, machineCost, skillCost, ga_params=ga_params))
    print(optimizeSkillAssignment(n_servers, failureRates, serviceRates, holdingCosts, penalty, machineCost, skillCost, ga_params=ga_params))


def test3():
    #test consistency of execution time
    initDatabase("db_simulations",flag='r')

    for i in range(10):
        startTime = time.time()
        with open("simulated_sets_version2.json", "r") as json_file:
            for line in json_file:
                json_case = json.loads(line)

                print('\r', json_case["caseID"],)
                sys.stdout.flush()

                if True: #json_case["caseID"]=="Case: 0123":
                    m_skills = json_case["SKU_number"]
                    n_servers = json_case["n_servers"]
                    failureRates = np.array(json_case["failure_rates"])
                    serviceRates = np.array(json_case["service_rates"])

                    holding_costs = np.ones(m_skills)*1.0
                    penalty_cost = 50.0
                    machine_cost = 1000
                    skill_cost = 100

                    population = 50
                    generation = 10
                    crossover = 0.3
                    mutation = 0.10
                    indpb=0.05

                    ga_params = {'population': population, 'generation': generation, 'crossover': crossover,
                                 'mutation': mutation, 'indpb': indpb, 'tournsize': 10} # tuned GA parameters

                    optimizeSkillAssignment(n_servers, failureRates, serviceRates, holding_costs, penalty_cost, machine_cost, skill_cost,
                                    ga_params = ga_params, runTheSimulation=False)

            json_file.close()
        print("Run ", i, "| Runtime: ", time.time() - startTime)

    pass

if __name__ == "__main__":
    startTime = time.time()

    test1()

    print("Total RunTime:", time.time() - startTime)
    # test()

