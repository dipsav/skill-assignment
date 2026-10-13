# =====================================================================================================
# SimulationInterface_fast.py -- SimulationInterface.py with the compiled simulator and two guards (3 October 2026).
#
#   SIMULATOR: simulation_batch_run from simulation_fast.py (Numba), same rules as simulation_pso_paper_v2.py;
#              FCFS when no priorities are given. maxQueue default raised to 2000 (GUARD 2 catches truncation).
#   CHECK:     checkSkillAssignment from simulation_fast.py (same LP as your original, solved in-process with scipy/HiGHS).
#
#   GUARD 1 (in simulation_optimization_run): checkSkillAssignment (simulation_fast.py: SKUs nobody can repair, then the LP
#            of constraints 2-8 with load <= 0.999) runs once, BEFORE the cache lookup. In the original interface it
#            ran only on a cache miss, which let a cached infeasible design through (case 0001 of the 2019 PSO runs).
#   GUARD 2 (in OptimizeStockLevelsAndCostsSimBased): if a SKU's simulated distribution does not sum to ~1, part
#            of it lies beyond maxQueue (a queue growing without bound, or maxQueue too small); the cost is then
#            infinite instead of being computed from the truncated part, which made such designs look cheap.
# Everything else is unchanged.
# =====================================================================================================
from __future__ import division
import json
# import shelve
import time
import numpy as np
import multiprocessing as mp

# from QueueingModels import OptimizeStockLevelsAndCosts
from simulation_codes.simulation_fast import simulation_batch_run, simulation, checkSkillAssignment   # compiled simulator + check
from simulation_codes.simulation_fast import checkSkillAssignment as _lp_check
min_coverage = 1         # B6b (10 Oct 2026): a skill matrix is feasible only if every SKU has at least this many technicians


def checkSkillAssignment(assignement, lambdas, mus, cap=0.999):
    """The LP feasibility check of simulation_fast, plus the coverage requirement min_coverage (1 = none)."""
    if min_coverage > 1 and np.any(np.asarray(assignement).sum(axis=0) < min_coverage):
        return False, None
    return _lp_check(assignement, lambdas, mus, cap)
import simulation_codes.filedict as filedict             # your on-disk cache, unchanged
import os
from scipy.optimize import linprog

MASS_TOLERANCE = 1e-3      # GUARD 2: a distribution must sum to at least 1 - MASS_TOLERANCE

# (the former lp_feasible is replaced by checkSkillAssignment from simulation_fast.py: one implementation)


database = None

stopNrRequestsDefault = 100000
replicationsDefault = 35
maxQueue = 2000
rounding_decimals = 5
nCores = None  # maximum number of cores will be used
# Repair-priority rule applied to every simulation of this process: per-SKU values, lower = higher priority (e.g. 1/mu
# for the mu rule); None = FCFS. Set it before a search (rerun_pso_cases.py --rule). The cache key does not include it,
# so a database must not be shared between rules (the driver keeps one in-memory cache per case and process).
priorities = None
preemptive = True        # False: non-preemptive priorities (A3, 10 Oct 2026)
# Search budget (benchmark C1): the number of DISTINCT designs a search may simulate. Counted as the designs stored in the
# cache (useDatabase=True; infeasible designs are never simulated or stored). None = no limit (default; all runs so far).
budget = None
simulated = 0


class BudgetSpent(Exception):
    """Raised before a simulation that would exceed SIf.budget; the optimizers catch it and return their best design."""

def initDatabase(databaseFile = "db_simulations", createNewDatabase = False):
    filename = databaseFile + ".dict"
    if createNewDatabase and os.path.isfile(filename):
        # try to remove old database
        os.remove(filename)
        pass

    global database
    # possible flags are:
    #         'r'	 - Open existing database for reading only (default)
    #         'w'	 - Open existing database for reading and writing
    #         'c'	 - Open database for reading and writing, creating it if it doesn't exist (default)
    #         'n'	 - Always create a new, empty database, open for reading and writing

    # database = shelve.open(databaseFile, protocol=2, flag=flag, writeback=True)
    database = filedict.FileDict(filename = filename)

    pass


# DONE: TODO create one function writing to DB
# DONE: TODO create key forming function
# TODO normalization, currently not essential as all experiments assume normalized lambdas
# DONE: TODO round failure and service rates to 5 digits after comma
# DONE: TODO create sorting and unsorting functionality by the failure rates


def makeCaseKey(n_servers, m_skills, failureRates, serviceRates):
    # str1 = "N:"+str(n_servers)+"|M:" + str(m_skills) + '|FR:[ '+str('  '.join(map(str, failureRates)))+']|SR:[ '+str('  '.join(map(str, serviceRates)))+']'
    # str2 = "N:"+str(n_servers)+"|M:" + str(m_skills) + "|FR:"+str(failureRates) + "|SR:" + str(serviceRates)
    #
    # print str1
    # print str2


    return "N:"+str(n_servers)+"|M:" + str(m_skills) + '|FR:['+str(','.join(map(str, failureRates)))+']|SR:['+str(','.join(map(str, serviceRates)))+']'
    # return "N:"+str(n_servers)+"|M:" + str(m_skills) + "|FR:"+str(failureRates) + "|SR:" + str(serviceRates)

def makeCaseKey2(n_servers, m_skills, failureRates, serviceRates, assignment):
    # str1 = "N:"+str(n_servers)+"|M:" + str(m_skills) + '|FR:[ '+str('  '.join(map(str, failureRates)))+']|SR:[ '+str('  '.join(map(str, serviceRates)))+']'
    # str2 = "N:"+str(n_servers)+"|M:" + str(m_skills) + "|FR:"+str(failureRates) + "|SR:" + str(serviceRates)
    #
    # print str1
    # print str2


    return "N:"+str(n_servers)+"|M:" + str(m_skills) + '|FR:['+str(','.join(map(str, failureRates)))+']|SR:['+str(','.join(map(str, serviceRates)))+']'\
           +'|Ass:['+str(','.join(map(str, ['['+str(','.join(map(str, x)))+']' for x in assignment])))+']'


def writeToDB1(n_servers, m_skills, failureRates, serviceRates, skillServerAssignment, numberInSystemDistributionAverage, utilizationRatesAverage, skillServerAssignmentStatisitcsAverage):
    sorting_order = failureRates.argsort()
    sorting_dict = dict(zip(sorting_order.tolist(), range(m_skills)))
    reverse_order = [sorting_dict[i] for i in sorted(sorting_dict.keys())]  # map(sorting_dict.get, sorting_dict.keys().sort())

    sorted_failureRates = np.round(failureRates[sorting_order], rounding_decimals)
    sorted_serviceRates = np.round(serviceRates[sorting_order], rounding_decimals)
    sorted_assignment = skillServerAssignment[:,sorting_order] # sorted on skills based on increasing failure rates

    queue_characherisitcs_key = makeCaseKey(n_servers, m_skills, sorted_failureRates, sorted_serviceRates )

    case = None
    databaseUpdated = False

    if queue_characherisitcs_key in database:
        case = database[queue_characherisitcs_key]
    else:
        case = {}
        case["m_skills"] = m_skills
        case["n_servers"] = n_servers
        case["failureRates"] = sorted_failureRates
        case["serviceRates"] = sorted_serviceRates
        case["skills_assignments"] = {}
        database[queue_characherisitcs_key] = case

    skillServerAssignment_key = str(sorted(sorted_assignment.tolist()))  # the assignment is sorted on server

    if not skillServerAssignment_key in case["skills_assignments"]:

        case["skills_assignments"][skillServerAssignment_key] = {}
        maxQueue = sum(np.sum(numberInSystemDistributionAverage,axis=0)>0)
        case["skills_assignments"][skillServerAssignment_key]["maxQueue"] = maxQueue
        case["skills_assignments"][skillServerAssignment_key]["skill_assignment"] = np.array(sorted(sorted_assignment.tolist()))
        case["skills_assignments"][skillServerAssignment_key]["numberInSystemDistributionAverage"] = numberInSystemDistributionAverage[sorting_order,:maxQueue]
        # case[skillServerAssignment_key]["utilizationRatesAverage"] = utilizationRatesAverage
        # case[skillServerAssignment_key]["skillServerAssignmentStatisitcsAverage"] = skillServerAssignmentStatisitcsAverage

        database[queue_characherisitcs_key] = case
        # database.sync()    # database updated
        databaseUpdated = True
        # print "databaseUpdated: ", skillServerAssignment.tolist()


    return databaseUpdated

def writeToDB2(n_servers, m_skills, failureRates, serviceRates, skillServerAssignment, numberInSystemDistributionAverage, utilizationRatesAverage, skillServerAssignmentStatisitcsAverage):
    '''
    In this function for each skill assignment a separate record in the database will be created
    '''

    sorting_order = failureRates.argsort()

    sorted_failureRates = np.round(failureRates[sorting_order], rounding_decimals)
    sorted_serviceRates = np.round(serviceRates[sorting_order], rounding_decimals)
    sorted_assignment = skillServerAssignment[:, sorting_order] # sorted on skills based on increasing failure rates

    skillServerAssignment_key = sorted(sorted_assignment.tolist())  # the assignment is sorted on server
    queue_assignment_characherisitcs_key = makeCaseKey2(n_servers, m_skills, sorted_failureRates, sorted_serviceRates, skillServerAssignment_key)

    databaseUpdated = False
    # print(queue_assignment_characherisitcs_key, queue_assignment_characherisitcs_key in database)

    if not queue_assignment_characherisitcs_key in database:
        case = {}
        case["m_skills"] = m_skills
        case["n_servers"] = n_servers
        case["failureRates"] = sorted_failureRates
        case["serviceRates"] = sorted_serviceRates
        case["skills_assignments"] = np.array(sorted(sorted_assignment.tolist()))

        maxQueue = sum(np.sum(numberInSystemDistributionAverage,axis=0)>0)
        case["maxQueue"] = maxQueue
        case["numberInSystemDistributionAverage"] = numberInSystemDistributionAverage[sorting_order,:maxQueue]
        case["utilizationRatesAverage"] = utilizationRatesAverage
        case["skillServerAssignmentStatisitcsAverage"] = skillServerAssignmentStatisitcsAverage

        database[queue_assignment_characherisitcs_key] = case
        databaseUpdated = True

        # print(queue_assignment_characherisitcs_key)

    return databaseUpdated

def getRecordFromDB1(n_servers, m_skills, failureRates, serviceRates, skillServerAssignment):
    sorting_order = failureRates.argsort()
    sorting_dict = dict(zip(sorting_order.tolist(), range(m_skills)))
    reverse_order = [sorting_dict[i] for i in sorted(sorting_dict.keys())]  # map(sorting_dict.get, sorting_dict.keys().sort())

    sorted_failureRates = np.round(failureRates[sorting_order], rounding_decimals)
    sorted_serviceRates = np.round(serviceRates[sorting_order], rounding_decimals)
    sorted_assignment = skillServerAssignment[:,sorting_order] # sorted on skills based on increasing failure rates

    queue_characherisitcs_key = makeCaseKey(n_servers, m_skills, sorted_failureRates, sorted_serviceRates )

    skillServerAssignment_key = str(sorted(sorted_assignment.tolist()))  # the assignment is sorted on server

    case = None
    if queue_characherisitcs_key in database:
        case = database[queue_characherisitcs_key]["skills_assignments"]

    if case: # case found
        if skillServerAssignment_key in case:
            serverAssignment = case[skillServerAssignment_key]
            return serverAssignment["numberInSystemDistributionAverage"][reverse_order]
        else:
            return None


def getRecordFromDB2(n_servers, m_skills, failureRates, serviceRates, skillServerAssignment):
    sorting_order = failureRates.argsort()
    sorting_dict = dict(zip(sorting_order.tolist(), range(m_skills)))
    reverse_order = [sorting_dict[i] for i in sorted(sorting_dict.keys())]  # map(sorting_dict.get, sorting_dict.keys().sort())

    sorted_failureRates = np.round(failureRates[sorting_order], rounding_decimals)
    sorted_serviceRates = np.round(serviceRates[sorting_order], rounding_decimals)
    sorted_assignment = skillServerAssignment[:,sorting_order] # sorted on skills based on increasing failure rates

    skillServerAssignment_key = sorted(sorted_assignment.tolist())  # the assignment is sorted on server
    queue_characherisitcs_key = makeCaseKey2(n_servers, m_skills, sorted_failureRates, sorted_serviceRates, skillServerAssignment_key )

    # keys_in_DB = list(database)

    # print(keys_in_DB[:10])

    # print(len(keys_in_DB), queue_characherisitcs_key in set(keys_in_DB), len(database), queue_characherisitcs_key in database, len(set(keys_in_DB)))


    if queue_characherisitcs_key in database:
        return database[queue_characherisitcs_key]["numberInSystemDistributionAverage"][reverse_order]
    else:
        return None

def OptimizeStockLevelsAndCostsSimBased(holdingCosts, penalty, marginalDistribution):

    if not isinstance(holdingCosts, np.ndarray):
        holdingCosts = np.array(holdingCosts)

    if len(marginalDistribution.shape) == 1:
        marginalDistribution = marginalDistribution.reshape(1,len(marginalDistribution))

    nSKUs=len(holdingCosts)
    if np.any(marginalDistribution.sum(axis=1) < 1 - MASS_TOLERANCE):            # GUARD 2
        return float('inf'), np.zeros(nSKUs, dtype=int), np.full(nSKUs, np.inf)
    maxQueue = marginalDistribution.shape[1]
    n_array = np.array(range(maxQueue))
    S = np.zeros(nSKUs, dtype=int)
    PBO = np.sum(marginalDistribution[:,1:], axis=1)
    EBO = np.sum(marginalDistribution*np.array(range(marginalDistribution.shape[1])), axis=1)

    hb_ratio = holdingCosts/penalty
    for sk in range(nSKUs):
        while S[sk]<maxQueue and np.sum(marginalDistribution[sk, S[sk]+1:]) > hb_ratio[sk]:
            S[sk] += 1
            PBO[sk] = np.sum(marginalDistribution[sk,S[sk]+1:]) # -= marginalDistribution[sk, S[sk]]
            EBO[sk] = np.sum(marginalDistribution[sk,S[sk]:]*n_array[:-S[sk]]) #-= PBO[sk]

    totalCost = np.sum(S*holdingCosts) + np.sum(penalty*EBO)

    return totalCost, S, EBO

# writeToDB = writeToDB1
# getRecordFromDB = getRecordFromDB1

writeToDB = writeToDB2
getRecordFromDB = getRecordFromDB2

# @profile
def simulation_optimization_run(failureRates, serviceRates, holdingCosts, penalty, skillServerAssignment,
                                stopNrRequests = stopNrRequestsDefault,
                                replications = replicationsDefault,
                                numberWarmingUpRequests = 0,
                                runTheSimulation=True, useDatabase=True, maxQ = maxQueue, startSeed = 10):
    # startSeed: replication i uses startSeed + 20 i. The search keeps the default (common random numbers for all
    # candidates); a final evaluation should use other seeds and useDatabase=False.

    numberInSystemDistributionAverage = None
    utilizationRatesAverage = None
    skillServerAssignmentStatisitcsAverage = None

    m_skills = skillServerAssignment.shape[1]
    n_servers = skillServerAssignment.shape[0]

    databaseUpdated = False
    # GUARD 1: the feasibility check runs once, here, BEFORE the cache lookup, so no design is costed without it
    if not checkSkillAssignment(skillServerAssignment, failureRates, serviceRates)[0]:
        return float('inf'), np.zeros(m_skills, dtype=int), np.zeros(m_skills), False, None, None
    if useDatabase:
        numberInSystemDistributionAverage = getRecordFromDB(n_servers, m_skills, failureRates, serviceRates, skillServerAssignment)

    if numberInSystemDistributionAverage is None and runTheSimulation:
        if useDatabase and budget is not None:
            if simulated >= budget:
                raise BudgetSpent
            globals()["simulated"] = simulated + 1

        # run the simulaiton
        # print "run simulation:", skillServerAssignment, failureRates, serviceRates
        if True:                                                   # already checked above (GUARD 1)
            # print skillServerAssignment.tolist(), failureRates, serviceRates,
            numberInSystemDistributionAverage, _, utilizationRatesAverage, skillServerAssignmentStatisitcsAverage, _\
                    = simulation_batch_run( failureRates = failureRates,
                                            serviceRates = serviceRates,
                                            skillServerAssignment = skillServerAssignment,
                                            histogramBucketsInput = [],
                                            # stopTimeInput=stopTime,
                                            expectedTotalNumberRequests = stopNrRequests,
                                            numberWarmingUpRequests = numberWarmingUpRequests,
                                            replications=replications,
                                            maxQueue=maxQ,
                                            nCores=nCores,
                                            priorities=priorities,
                                            startSeed=startSeed,
                                            preemptive=preemptive)

            # print(numberInSystemDistributionAverage)

            if useDatabase:
                databaseUpdated = writeToDB(n_servers, m_skills, failureRates, serviceRates, skillServerAssignment,
                                            numberInSystemDistributionAverage, utilizationRatesAverage, skillServerAssignmentStatisitcsAverage)


    totalCost = float('inf'); S = np.zeros(m_skills, dtype=int); EBO = np.zeros(m_skills)
    if numberInSystemDistributionAverage is not None:
        # print [np.nanmean([numberInSystemDistributionAverage[j][i-1]/numberInSystemDistributionAverage[j][i]*failureRates[j] for i in range(3,15)]) for j in range(m_skills)]
        totalCost, S, EBO = OptimizeStockLevelsAndCostsSimBased(holdingCosts, penalty, numberInSystemDistributionAverage)

    # print(databaseUpdated, totalCost)

    return totalCost, S, EBO, databaseUpdated, utilizationRatesAverage, skillServerAssignmentStatisitcsAverage


# def simulation_optimization_runV2(failureRates, serviceRates, holdingCosts, penalty, skillServerAssignment,
#                                 replications = replicationsDefault,
#                                 stopNrRequests = stopNrRequestsDefault,
#                                 numberWarmingUpRequests = 0, maxQ = maxQueue):
#
#     totalCostList = []
#     for i in range(replications):
#         numberInSystemDistributionAverage, _, _, _ = \
#             simulation_batch_run(failureRates = failureRates,
#                                 serviceRates = serviceRates,
#                                 skillServerAssignment = skillServerAssignment,
#                                 histogramBucketsInput = [],
#                                 expectedTotalNumberRequests = stopNrRequests,
#                                 numberWarmingUpRequests = numberWarmingUpRequests,
#                                 replications=1,
#                                 maxQueue=maxQ,
#                                 startSeed=100+i*30)
#
#
#         if numberInSystemDistributionAverage is not None:
#             totalCost, _, _ = OptimizeStockLevelsAndCostsSimBased(holdingCosts, penalty, numberInSystemDistributionAverage)
#             totalCostList.append(totalCost)
#
#     return totalCostList


# def simulation_optimization_batchrun(failureRates, serviceRates, holdingCosts, penalty, skillServerAssignment,
#                                   replications=replicationsDefault,
#                                   stopNrRequests = stopNrRequestsDefault,
#                                   numberWarmingUpRequests=0, maxQ=maxQueue):
#     totalCostList = []
#
#     pool = mp.Pool(processes=nCores)
#
#     requestArrivalRateInput = np.sum(failureRates)
#     skillDistributionInput = np.cumsum(failureRates/requestArrivalRateInput)
#     # skillServerCosts = np.ones(skillServerAssignment.shape)
#
#     skillServerRates = np.transpose(np.zeros(skillServerAssignment.shape))
#     for i in range(serviceRates.shape[0]): skillServerRates[i,:] = skillServerAssignment[:,i] * serviceRates[i]
#     skillServerRates = np.transpose(skillServerRates)
#     skillServerCosts = skillServerRates
#
#     startSeed = 50
#     try:
#         results = [pool.apply_async(simulation,
#                      args=(requestArrivalRateInput,
#                         skillDistributionInput,  #The distribution should be in cummulative form
#                         skillServerRates, ##replenishment times can be different for different priority classes
#                         skillServerCosts,
#                         [],
#                         stopNrRequests,
#                         numberWarmingUpRequests,
#                         startSeed+20*i,
#                         ) )
#             for i in range(replications)]
#
#         for p in results:
#             numberInSystemDistribution = p.get()[0]
#
#             numberInSystemDistributionArray = np.zeros((len(failureRates), maxQueue))
#             for sk in numberInSystemDistribution:
#                 for n in numberInSystemDistribution[sk]:
#                     if(n<maxQ):
#                         numberInSystemDistributionArray[sk,n] += numberInSystemDistribution[sk][n]
#
#             totalCost, _, _ = OptimizeStockLevelsAndCostsSimBased(holdingCosts, penalty, numberInSystemDistributionArray)
#             totalCostList.append(totalCost)
#
#         #print "Runs are executed:", len(results)
#         pool.close()
#     except KeyboardInterrupt:
#         print('got ^C while pool mapping, terminating the pool')
#         pool.terminate()
#         print('pool is terminated')
#     except Exception as e:
#         print('got exception: %r, terminating the pool' % (e,))
#         pool.terminate()
#         print('pool is terminated')
#     finally:
#         pool.join()
#
#     return totalCostList

def createDatabaseFromExistingJSON(json_file, database_file):
    file_path = json_file  # "..//simulated_sets_version4.json"

    initDatabase(databaseFile=database_file, createNewDatabase=True)

    with open(file_path, "r") as json_file:
        json_file.readline()
        for line in json_file:
            json_case = json.loads(line)

            if True: #json_case["caseID"] == "Case: 0061": #debuging line
                m_skills = json_case["n_classes"]
                n_servers = json_case["m_servers"]
                failureRates = np.array(json_case["arrival_rates"])
                serviceRates = np.array(json_case["service_rates"])

                for assignment in json_case["assignments"].values():
                    skillServerAssignment = np.array(assignment["class_server_assignment"]) #map(list, zip(*assignment["skill_assignment"]))) # transpose the assignment

                    #maxQueue = max(len(val) for val in assignment["queue_length_distributions"].values())
                    maxQueue = max([max([int(val) for val in queue_req.keys()]) for queue_req in assignment["queue_length_distributions"].values()]) + 1
                    marginalDistribution = np.zeros((len(assignment["queue_length_distributions"]), maxQueue))
                    for sk in assignment["queue_length_distributions"]:
                        for n in assignment["queue_length_distributions"][sk]:
                            marginalDistribution[int(sk), int(n)] = assignment["queue_length_distributions"][sk][n]

                    upt = writeToDB(n_servers, m_skills, failureRates, serviceRates, skillServerAssignment, marginalDistribution, assignment["utilization_rates"], assignment["class_server_distribution"])
                    pass

            print(json_case["caseID"])

            # if json_case["caseID"] == "Case: 0010":
            #     break

    # database.close()
    pass



if __name__ == '__main__':
    np.set_printoptions(linewidth=50000)

    startTime = time.time()
    createDatabaseFromExistingJSON("..//simulated_sets_version4.json", "..//db_simulations_2019")



    # # testDatabaseInterface()
    # #
    # # failure_rates = np.array([  2.23738889e-01,   4.94087095e-02,   9.83533638e-03,   2.32382124e-01,   3.31158691e-05,   7.81429552e-02,   2.04179403e-01,   1.43352018e-01,   4.02415285e-02,   1.86859200e-02])
    # # service_rates = np.array([  1.76753604e+00,   4.69646045e-02,   2.32412359e+00,   3.29198749e-01,   1.99622149e-05,   5.00131046e-01,   4.21953234e-01,   7.21777996e-02,   5.78270470e-02,   1.65334582e-02])
    # # fullflex_assignment = np.ones((10,10), dtype=int)
    # #
    # # holding_costs = np.array([ 543.85807009,  577.29253873,  817.43195388,  688.21756066,  114.2328496 ,  166.58437791,  910.89245915,  598.73244586,  638.60568814,  901.15711987])
    # # penalty_cost = 32277.41185467891
    # #
    # failure_rates = np.array([0.2541783360098542, 0.31538730923198566, 0.9915311240730542, 0.13692735636243197, 0.10197587432267417])
    # service_rates = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    # holding_cost = np.array([386.4200669605343, 264.8119394757236, 53.76775526966686, 775.1555433515449, 1000.0])
    #
    # penalty_cost_factor = 50
    # penalty = penalty_cost_factor*sum(failure_rates*holding_cost)/sum(failure_rates)  #24801.5530506
    #
    # assignment1    = np.array([[0, 1, 1, 0, 1], [1, 0, 1, 1, 0]])
    # assignment2    = np.array([[0, 0, 1, 0, 0], [1, 1, 1, 1, 1]])
    #
    # initDatabase(databaseFile = "db_simulations_test", createNewDatabase = True)
    #
    # totalCost, S, EBO, databaseUpdated, utilizationRatesAverage, skillServerAssignmentStatisitcsAverage = \
    #     simulation_optimization_run(failure_rates, service_rates, holding_cost, penalty, assignment2,
    #                                   numberWarmingUpRequests = 0, #5000,
    #                                   stopNrRequests = 1000, #1000000,
    #                                   replications = 1, #50,
    #                                   runTheSimulation = True, useDatabase = True)
    #     # simulation_optimization_run(failure_rates, service_rates, holding_costs, penalty_cost, fullflex_assignment, runTheSimulation=True, useDatabase=False)
    #
    #
    #
    # # totalCostList = \
    # #     simulation_optimization_bathrun(failure_rates, service_rates, holding_cost, penalty, assignment2,
    # #                                   numberWarmingUpRequests = 5000,
    # #                                   stopNrRequests = 1000000,
    # #                                   replications = 35)
    #
    #
    # print(totalCost)
    # print(S*holding_cost + penalty*EBO)
    # print(utilizationRatesAverage)
    # print(skillServerAssignmentStatisitcsAverage)
    # print(databaseUpdated)

    print(time.time() - startTime)
    pass