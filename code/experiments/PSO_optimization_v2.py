# =====================================================================================================
# PSO_optimization_v2.py -- PSO_optimization.py with the issues found in the revision fixed (3 October 2026).
# Every change is marked "# CHANGE n". The search itself (particles, velocity update V4, inertia schedule) is unchanged.
#
#   CHANGE 1  Evaluations go through SimulationInterface_fast: feasibility checked before the cache lookup, distributions
#             up to 2000 with an infinite cost if one is cut off, the compiled simulator.
#   CHANGE 2  The `seed` argument is used (np.random and random); it was accepted but ignored, so runs were not reproducible.
#   CHANGE 3  np.alltrue (removed in NumPy 2) -> np.all.
#   CHANGE 4  The last generation counts: the loop moved and evaluated the particles after its final best-update, so the
#             last generation's evaluations were never compared with the bests. One more update after the loop.
#   CHANGE 5  The reported cost of the best design comes from an INDEPENDENT evaluation (35 replications, seeds
#             final_seed + 20 i, no cache), not from the seeds that selected it (those make the reported cost optimistic).
#             The utilization and assignment statistics returned are this design's own. The search cost is kept as
#             logbook.search_total_cost.
# Unchanged on purpose: the velocity update on contested bits multiplies the velocity by inertia*(r2 - c2/2)
# (the code comment says "+"; the paper text will describe the multiplication); initial particles; fitness tuple.
# =====================================================================================================
# coding=utf-8
import numpy as np

import itertools

from deap import base
from deap import creator
from deap import tools

from tqdm import trange


# import sys
# sys.path.insert(0, '/Users/andrei/Documents/Research/Research Papers/Skill Assignment/PythonCodes/')
#
from simulation_codes.SimulationInterface_fast import simulation_optimization_run, initDatabase   # CHANGE 1
from simulation_codes import SimulationInterface_fast as SIf
import random
import time


def updateParticleV1(part, best, c1=1.0, c2=1.0, intertia=1.0):

    #   the velocity is updated as:
    #   V_ij(t + 1) = w_t × V_ij(t)                 when pBest_ij <> gBest_j
    #   V_ij(t + 1) = w_t × V_ij(t) +c1r1 +c2r2     when pBest_ij == gBest_j == 1
    #   V_ij(t + 1) = w_t × V_ij(t) -c1r1 -c2r2     when pBest_ij == gBest_j == 0
    u1 = np.random.uniform(0, c1)
    u2 = np.random.uniform(0, c2)
    part.speed =  part.speed*intertia \
                  +  (u1 + u2)*(np.array(part.best)==np.array(best))*(2*np.array(part.best)-1)

    try:
        sigmoid = 1/(1+np.exp(-part.speed))
    except RuntimeWarning:
        print( part.speed )

    r = np.random.uniform(0,1,np.array(part).shape)

    new_assignment = ((r< sigmoid)*1).tolist()
    updated = part[:] != new_assignment
    part[:] = new_assignment

    return updated

def updateParticleV1a(part, best, c=1.0, intertia=1.0):

    #   the velocity is updated as:
    #   V_ij(t + 1) = w_t × V_ij(t)           when pBest_ij <> gBest_j
    #   V_ij(t + 1) = w_t × V_ij(t) + c*r     when pBest_ij == gBest_j == 1
    #   V_ij(t + 1) = w_t × V_ij(t) - c*r     when pBest_ij == gBest_j == 0
    u = np.random.uniform(0, c)
    part.speed =  part.speed*intertia +  u*(np.array(part.best)==np.array(best))*(2*np.array(part.best)-1)

    try:
        sigmoid = 1/(1+np.exp(-part.speed))
    except RuntimeWarning:
        print( part.speed )

    r = np.random.uniform(0,1,np.array(part).shape)

    new_assignment = ((r< sigmoid)*1).tolist()
    updated = part[:] != new_assignment
    part[:] = new_assignment

    return updated

def updateParticleV2(part, best, c1=1.0, c2=1.0, intertia=1.0):

    #   the velocity is updated as:
    #
    #
    #
    u1 = np.random.uniform(0, c1)
    u2 = np.random.uniform(0, c2)


    d11 = np.zeros(np.array(part).shape)
    d01 = np.zeros(np.array(part).shape)
    d12 = np.zeros(np.array(part).shape)
    d02 = np.zeros(np.array(part).shape)

    n = np.array(part).shape[0]
    m = np.array(part).shape[1]

    for i in range(n):
        for j in range(m):
            if np.array(part.best)[i,j]:
                d01[i,j] = -u1
                d11[i,j] = u1
            else:
                d01[i,j] = u1
                d11[i,j] = -u1

            if np.array(best)[i,j]:
                d02[i,j] = -u2
                d12[i,j] = u2
            else:
                d02[i,j] = u2
                d12[i,j] = -u2

    part.speed0 =  part.speed0*intertia + d01 + d02 # u1*(1-2*np.array(part.best)) + u2*(1-2*np.array(best))
    part.speed1 =  part.speed1*intertia + d11 + d12 # u1*(2*np.array(part.best)-1) + u2*(2*np.array(best)-1)

    # part.speed =  part.speed0*np.array(part) + part.speed1*(1-np.array(part))


    # $V_{ij}(t) = wV_{ij}(t) + r_1 c_1 (1-2P_{ij}^{best}) + r_2 c_2 (1-2P_{ij}^{gbest})$
    # $sig(V_{ij}(t)) = \frac{1}{1 + e^{-V_{ij}(t)\cdot(2x_{ij}-1)}}$
    # new option
    part.speed =  part.speed*intertia + u1*(1-2*np.array(part.best)) + u2*(1-2*np.array(best))

    try:
        # sigmoid = 1/(1+np.exp(-(part.speed0*np.array(part) + part.speed1*(1-np.array(part)))))
        sigmoid = 1 / (1 + np.exp(-(part.speed*(2*np.array(part)-1)) ))
    except RuntimeWarning:
        print( part.speed )

    r = np.random.uniform(0,1,np.array(part).shape)

    new_assignment = ((r< sigmoid)*1).tolist()
    updated = part[:] != new_assignment
    part[:] = new_assignment

    return updated

def updateParticleV3(part, best, c1=1.0, c2=1.0, intertia=1.0):

    #   the velocity is updated as:
    #   V_ij(t + 1) = w_t × V_ij(t) + c2*r2-c2/2  when pBest_ij <> gBest_j
    #   V_ij(t + 1) = w_t × V_ij(t) + c1*r1       when pBest_ij == gBest_j == 1
    #   V_ij(t + 1) = w_t × V_ij(t) - c1*r1       when pBest_ij == gBest_j == 0
    u1 = np.random.uniform(0, c1, np.array(part).shape)
    u2 = np.random.uniform(0, c2, np.array(part).shape)
    part.speed =  part.speed*intertia \
                  +  u1*(np.array(part.best)==np.array(best))*(2*np.array(part.best)-1) \
                  +  (u2 - c2/2)*(np.array(part.best)!=np.array(best))

    try:
        sigmoid = 1/(1+np.exp(-part.speed))
    except RuntimeWarning:
        print( part.speed )

    r = np.random.uniform(0,1,np.array(part).shape)

    new_assignment = ((r< sigmoid)*1).tolist()
    updated = part[:] != new_assignment
    part[:] = new_assignment

    return updated

def updateParticleV4(part, best, c1=1.0, c2=1.0, intertia=1.0):

    #   the velocity is updated as:
    #   V_ij(t + 1) = V_ij(t) + c2*r2-c2/2  when pBest_ij <> gBest_j
    #   V_ij(t + 1) = w_t × V_ij(t) + c1*r1       when pBest_ij == gBest_j == 1
    #   V_ij(t + 1) = w_t × V_ij(t) - c1*r1       when pBest_ij == gBest_j == 0
    u1 = np.random.uniform(0, c1, np.array(part).shape)
    u2 = np.random.uniform(0, c2, np.array(part).shape)
    part.speed =  part.speed*intertia *((np.array(part.best)==np.array(best)) + (u2 - c2/2)*(np.array(part.best)!=np.array(best)))\
                  +  u1*(np.array(part.best)==np.array(best))*(2*np.array(part.best)-1)

    try:
        sigmoid = 1/(1+np.exp(-part.speed))
    except RuntimeWarning:
        print( part.speed )

    r = np.random.uniform(0,1,np.array(part).shape)

    new_assignment = ((r< sigmoid)*1).tolist()
    updated = part[:] != new_assignment
    part[:] = new_assignment

    return updated

def updateParticleOriginal(part, best, c1=1.0, c2=1.0, intertia=1.0):
    """The BPSO of El-Maleh et al. (2013) as used in Sleptchenko et al. (ICACI 2017), Eq. (13):
    V_ij(t+1) = w_t V_ij(t) + (c1 r1 + c2 r2) (2 B^p_ij - 1) I(B^p_ij = B^g_ij), r1, r2 ~ U[0, 1] per element;
    the inertia decreases (Eq. 14, set in the main loop); bits drawn through the sigmoid."""
    shape = np.array(part).shape
    r1 = np.random.uniform(0, 1, shape); r2 = np.random.uniform(0, 1, shape)
    agree = (np.array(part.best) == np.array(best))
    part.speed = part.speed * intertia + (c1 * r1 + c2 * r2) * (2 * np.array(part.best) - 1) * agree
    sigmoid = 1 / (1 + np.exp(-part.speed))
    r = np.random.uniform(0, 1, shape)
    new_assignment = ((r < sigmoid) * 1).tolist()
    updated = part[:] != new_assignment
    part[:] = new_assignment
    return updated


def updateParticleReset(part, best, c1=1.0, c2=1.0, intertia=1.0):
    """Section 4.1 as written (Eq. 12): agreed bits w V +/- c1 r1; contested bits RESET to c2 (r2 - 1/2), independent of
    the inertia and of the old velocity (the code's V4 multiplies the old velocity by w c2 (r2 - 1/2) instead)."""
    shape = np.array(part).shape
    u1 = np.random.uniform(0, c1, shape); u2 = np.random.uniform(0, c2, shape)
    agree = (np.array(part.best) == np.array(best))
    part.speed = (part.speed * intertia + u1 * (2 * np.array(part.best) - 1)) * agree + (u2 - c2 / 2) * (~agree)
    sigmoid = 1 / (1 + np.exp(-part.speed))
    r = np.random.uniform(0, 1, shape)
    new_assignment = ((r < sigmoid) * 1).tolist()
    updated = part[:] != new_assignment
    part[:] = new_assignment
    return updated


TRACK = {"cost": float("inf"), "X": None}       # best design evaluated in the current search (used if the budget stops it)
PARALLEL_WORKERS = 1                              # > 1: the particles that moved in a generation are simulated in parallel


def _sim_worker(args):
    """Simulate one design (search seeds) in a worker process; returns the cache record for the parent's database."""
    X, lam, mu, h, b, pr, pre = args
    SIf.priorities = pr; SIf.preemptive = pre
    dist, _, util, assign, _ = SIf.simulation_batch_run(failureRates=lam, serviceRates=mu, skillServerAssignment=X, histogramBucketsInput=[],
                                                        expectedTotalNumberRequests=100000, numberWarmingUpRequests=0, replications=35,
                                                        maxQueue=SIf.maxQueue, nCores=1, startSeed=10)
    return X, dist, util, assign


def _evaluate_parallel(parts, toolbox, lam, mu, h, b, f, cs):
    """Same result as evaluating the moved particles one by one: designs not yet in the cache (and feasible) are simulated
    in a process pool, written to the cache, and every particle is then costed through the normal path (cache hits).
    The budget counts the distinct designs simulated, as in the serial path."""
    from multiprocessing import Pool
    N = len(lam); todo = {}
    for part in parts:
        X = np.array(part).reshape(-1, N)
        if SIf.getRecordFromDB(X.shape[0], N, lam, mu, X) is None and SIf.checkSkillAssignment(X, lam, mu)[0]:
            todo.setdefault(X.tobytes(), X)
    todo = list(todo.values())
    if SIf.budget is not None and SIf.simulated + len(todo) > SIf.budget:
        todo = todo[:max(0, SIf.budget - SIf.simulated)]            # the rest hit the budget below, as in the serial path
    if todo:
        with Pool(min(PARALLEL_WORKERS, len(todo))) as pool:
            for X, dist, util, assign in pool.map(_sim_worker, [(X, lam, mu, h, b, SIf.priorities, SIf.preemptive) for X in todo]):
                SIf.writeToDB(X.shape[0], N, lam, mu, X, dist, util, assign)
                SIf.simulated += 1
    for part in parts:
        part.fitness.values = toolbox.evaluate(part)                  # cache hits, or BudgetSpent for designs left out


def createRandomAssignment(lambdas, mus, n_servers):
    assignment = []
    n = np.array(np.ceil(np.array(lambdas)/mus), dtype=int)
    m_skills = len(lambdas)

    for i in range(m_skills):
        # n_i = np.random.random_integers(int(n[i]), n_servers)
        n_i = min(max(n[i], SIf.min_coverage), n_servers)          # B6b: at least min_coverage technicians per SKU
        if n_i < n_servers:
            n_i = np.random.randint(n_i, n_servers)
        assignment.append(np.random.permutation(np.array([1]*n_i + [0]*(n_servers-n_i))).tolist())

    assignment = map(list, zip(*assignment)) #transpose the obtained list
    # print assignment

    return assignment # sum(assignment, []) # the sum flattens the list of lists



def generate(lambdas, mus, n_servers, speedRange = (-1000,1000)):
    part = creator.Particle(createRandomAssignment(lambdas, mus, n_servers))
    # part = creator.Particle([[1,1],[1,1]])
    part.speed  = np.random.uniform(speedRange[0], speedRange[1], (n_servers, len(lambdas))) #.tolist()
    part.speed0 = np.random.uniform(speedRange[0], speedRange[1], (n_servers, len(lambdas))) #.tolist()
    part.speed1 = np.random.uniform(speedRange[0], speedRange[1], (n_servers, len(lambdas))) #.tolist()
    return part


def evaluateParticleCost(individual, failure_rates, service_rates, holding_costs, penalty_cost, machine_cost, skill_cost, runTheSimulation=True):
    '''
    :param individual: the server assignment of the tested individual
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

    assignment = np.array(individual).reshape(-1, m_skills)  # converts into matrix form if assignment is a list
    totalCost, stock, ebo, databaseUpdated, _, _ = simulation_optimization_run(failure_rates, service_rates, holding_costs, penalty_cost, assignment, runTheSimulation=runTheSimulation, stopNrRequests=100000, replications=35)
    totalCost += skill_cost*np.sum(assignment)
    totalCost +=  machine_cost*sum((np.sum(assignment,axis=1)>0))
    if totalCost < TRACK["cost"]:
        TRACK["cost"], TRACK["X"] = totalCost, assignment.copy()
        TRACK.setdefault("trace", []).append((SIf.simulated, float(totalCost)))

    stock_cost = sum(stock * holding_costs)
    ebo_cost = sum(ebo*penalty_cost)
    servers_cost = machine_cost*sum((np.sum(assignment,axis=1)>0))
    skills_cost = skill_cost*np.sum(assignment)

    return totalCost, stock_cost, ebo_cost, servers_cost, skills_cost  # None, databaseUpdated


def convertQuantum(swarm, rcloud, centre, dist):
    dim = len(swarm[0])
    for part in swarm:
        position = [np.random.gauss(0, 1) for _ in range(dim)]
        disp = np.sqrt(sum(x**2 for x in position))

        if dist == "gaussian":
            u = abs(np.random.gauss(0, 1.0/3.0))
            part[:] = [(rcloud * x * u**(1.0/dim) / disp) + c for x, c in zip(position, centre)]

        elif dist == "uvd":
            u = np.random.random()
            part[:] = [(rcloud * x * u**(1.0/dim) / disp) + c for x, c in zip(position, centre)]

        elif dist == "nuvd":
            u = abs(np.random.gauss(0, 1.0/3.0))
            part[:] = [(rcloud * x * u / disp) + c for x, c in zip(position, centre)]

        del part.fitness.values
        del part.bestfit.values
        part.best = None

    return swarm

def optimizeSkillAssignmentPSO(n_servers, failure_rates, service_rates, holding_costs, penalty_cost, machine_cost, skill_cost, caseID='',
                            pso_params = {'generation': 10, 'population':5, 'c1': 1.0, 'c2': 1.0, 'inertia_max': 10.0, 'inertia_min': 10.0, 'speedRange': 0.0},
                            #ga_params = {'population': 100, 'generation': 50, 'crossover': 0.4, 'mutation': 0.1, 'indpb': 0.1, 'tournsize': 10},
                            seed = None, verbose = False, showProgressBar = False, runTheSimulation=True,
                            final_seed = 1_000_000):

    if seed is not None:                                                    # CHANGE 2
        np.random.seed(seed); random.seed(seed)
    old_settings = np.seterr(all='ignore', invalid='ignore')


    try:
        type(creator.FitnessMin)
    except Exception:
        creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0, -1.0, -1.0, -1.0)) #-1 for minimization
        # creator.create("Particle", np.ndarray, fitness=creator.FitnessMax, speed=np.ndarray, best=None)
        creator.create("Particle", list, fitness=creator.FitnessMin, speed=list, speed0=None, speed1=None, best=None)

    toolbox = base.Toolbox()
    toolbox.register("particle", generate, lambdas=failure_rates, mus=service_rates, n_servers=n_servers, speedRange = (-pso_params["speedRange"], pso_params["speedRange"]))
    toolbox.register("population", tools.initRepeat, list, toolbox.particle)
    # toolbox.register("update", updateParticleV1, c1=pso_params['c1'], c2=pso_params['c2'])
    # toolbox.register("update", updateParticleV1a, c=pso_params['c1'])
    # toolbox.register("update", updateParticleV2, c1=pso_params['c1'], c2=pso_params['c2'])
    variant = pso_params.get("variant", "proposed")      # "original": El-Maleh update, decreasing inertia (C1); "reset": Eq. 12 as written
    update = {"original": updateParticleOriginal, "reset": updateParticleReset}.get(variant, updateParticleV4)
    toolbox.register("update", update, c1=pso_params['c1'], c2=pso_params['c2'])
    TRACK["cost"], TRACK["X"], TRACK["trace"] = float("inf"), None, []
    toolbox.register("evaluate", evaluateParticleCost,
                                 failure_rates = failure_rates,
                                 service_rates = service_rates,
                                 holding_costs = holding_costs,
                                 penalty_cost = penalty_cost,
                                 machine_cost = machine_cost,
                                 skill_cost = skill_cost,
                                 runTheSimulation = runTheSimulation)


    tc_stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stock_stats = tools.Statistics(lambda ind: ind.fitness.values[1])
    ebo_stats = tools.Statistics(lambda ind: ind.fitness.values[2])
    server_stats = tools.Statistics(lambda ind: ind.fitness.values[3])
    skills_stats = tools.Statistics(lambda ind: ind.fitness.values[4])

    GEN = pso_params['generation']
    best = None
    pop = toolbox.population(n=pso_params['population'])

    # overwrite a random individuum with the fullflex assignement, since some of full-flex assignments are very good
    # rnd_ind = np.random.random_integers(0, len(pop)-1)
    # pop[rnd_ind][:] = np.ones((n_servers, len(failure_rates)), dtype=int).tolist()


    budget_stopped = False
    try:
        for part in pop:
            part.fitness.values = toolbox.evaluate(part)
    except SIf.BudgetSpent:
        budget_stopped = True


    mstats = tools.MultiStatistics(total=tc_stats, stock_cost=stock_stats, ebo_cost=ebo_stats, server_cost=server_stats, skill_cost=skills_stats)

    mstats.register("avg", lambda x: np.mean(np.array(x)[~np.isinf(x)]) if not np.all(np.isinf(x))  else -1) #np.mean)
    # stats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("maxninf", lambda x: np.max(np.array(x)[~np.isinf(x)]) if not np.all(np.isinf(x))  else -1) #np.max)
    mstats.register("max", np.max)
    mstats.register("median", np.median)
    mstats.register("Q25", np.percentile, q=25)
    mstats.register("Q75", np.percentile, q=75)

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (mstats.fields if mstats else [])


    if not budget_stopped:
        record = mstats.compile(pop) if mstats else {}
        logbook.record(gen=0, nevals=len(pop), **record)


    for g in trange(0 if budget_stopped else GEN+1, desc=caseID, disable=not showProgressBar):
        for part in pop:
            # part.fitness.values = toolbox.evaluate(part)
            if part.best is None or part.best.fitness < part.fitness:
                part.best = creator.Particle(part)
                part.best.fitness.values = part.fitness.values
            if best is None or best.fitness < part.fitness:
                best = creator.Particle(part)
                best.fitness.values = part.fitness.values
        nevals = 0
        try:
            if PARALLEL_WORKERS > 1:                                       # evaluate the moved particles in parallel
                moved = []
                for part in pop:
                    progress = min(1.0, SIf.simulated / SIf.budget) if SIf.budget else g / GEN
                    if variant == "original":
                        inertia = pso_params['inertia_max'] - progress*(pso_params['inertia_max'] - pso_params['inertia_min'])
                    else:
                        inertia = pso_params['inertia_min'] + progress**pso_params.get('inertia_power', 1.0)*(pso_params['inertia_max'] - pso_params['inertia_min'])
                    last_inertia = inertia
                    if toolbox.update(part, best, intertia=inertia):
                        moved.append(part)
                _evaluate_parallel(moved, toolbox, failure_rates, service_rates, holding_costs, penalty_cost, machine_cost, skill_cost)
                nevals = len(moved)
                continue
            for part in pop:
                # Progress of the schedule: the share of the generations (as published) or, under an evaluation budget
                # (SIf.budget, benchmark C1), the share of the budget of distinct simulated designs already used -- so
                # the schedule completes exactly when the budget runs out, whatever the cache-hit rate (decision 6 Oct).
                progress = min(1.0, SIf.simulated / SIf.budget) if SIf.budget else g / GEN
                if variant == "original":
                    inertia = pso_params['inertia_max'] - progress*(pso_params['inertia_max'] - pso_params['inertia_min'])
                else:
                    # increasing inertia; inertia_power p > 1 rises slowly first (default 1: linear, unchanged)
                    frac = progress ** pso_params.get('inertia_power', 1.0)
                    inertia = pso_params['inertia_min'] + frac*(pso_params['inertia_max'] - pso_params['inertia_min'])
                last_inertia = inertia
                if toolbox.update(part, best, intertia=inertia):
                    part.fitness.values = toolbox.evaluate(part)
                    nevals += 1
        except SIf.BudgetSpent:
            budget_stopped = True
            break
            # toolbox.update(part, best, intertia=inertia)
            # part.fitness.values = toolbox.evaluate(part)

        # Gather all the fitnesses in one list and print the stats
        record = mstats.compile(pop) if mstats else {}
        logbook.record(gen=g, evals=len(pop), nevals=nevals, **record)
        if verbose:
            print(best.fitness.values, logbook.stream)


    # CHANGE 4: the last generation counts. Skipped when the budget stopped the search: particles moved but not yet
    # evaluated would carry their old fitness; TRACK holds the best design actually evaluated.
    for part in ([] if budget_stopped else pop):
        if part.best is None or part.best.fitness < part.fitness:
            part.best = creator.Particle(part)
            part.best.fitness.values = part.fitness.values
        if best is None or best.fitness < part.fitness:
            best = creator.Particle(part)
            best.fitness.values = part.fitness.values

    if budget_stopped:                                                       # budget spent: best design evaluated
        best_assignment = np.array(TRACK["X"]); logbook.search_total_cost = TRACK["cost"]
    else:
        best_assignment = np.array(best).reshape(-1, len(failure_rates))
        logbook.search_total_cost = best.fitness.values[0]                 # CHANGE 5: cost with the search's seeds
    logbook.budget_stopped = budget_stopped
    logbook.last_inertia = locals().get("last_inertia")

    totalCost, S, EBO, _, utilizationRatesAverage, skillServerAssignmentStatisitcsAverage =\
        simulation_optimization_run(failure_rates, service_rates, holding_costs, penalty_cost, best_assignment,
                                    runTheSimulation=runTheSimulation, useDatabase=False, startSeed=final_seed)   # CHANGE 5
    totalCost += skill_cost*np.sum(best_assignment)
    totalCost +=  machine_cost*sum((np.sum(best_assignment,axis=1)>0))

    np.seterr(**old_settings)

    del creator.FitnessMin
    del creator.Particle

    return best_assignment, totalCost, S, EBO, logbook, utilizationRatesAverage, skillServerAssignmentStatisitcsAverage


# def optimizeSkillAssignmentMultiPSO(n_servers, failure_rates, service_rates, holding_costs, penalty_cost, machine_cost, skill_cost,
#                             pso_params = {'generation': 10, 'population':5, 'c1': 1.0, 'c2': 1.0, 'inertia_max': 10.0, 'inertia_min': 10.0, 'speedRange': 0.0},
#                             #ga_params = {'population': 100, 'generation': 50, 'crossover': 0.4, 'mutation': 0.1, 'indpb': 0.1, 'tournsize': 10},
#                             seed = None, verbose = False, runTheSimulation=True):
#
#     old_settings = np.seterr(all='ignore', invalid='ignore')
#
#
#     creator.create("FitnessMinMPSO", base.Fitness, weights=(-1.0,)) #-1 for minimization
#     # creator.create("Particle", np.ndarray, fitness=creator.FitnessMax, speed=np.ndarray, best=None)
#     # creator.create("Particle", list, fitness=creator.FitnessMin, speed=list, best=None)
#
#     creator.create("ParticleMPSO", list, fitness=creator.FitnessMinMPSO, speed=list, best=None, bestfit=creator.FitnessMinMPSO)
#     creator.create("Swarm", list, best=None, bestfit=creator.FitnessMinMPSO)
#
#
#     toolbox = base.Toolbox()
#     toolbox.register("particle", generate, lambdas=failure_rates, mus=service_rates, n_servers=n_servers, speedRange = (-pso_params["speedRange"], pso_params["speedRange"]))
#     toolbox.register("swarm", tools.initRepeat, creator.Swarm, toolbox.particle)
#     # toolbox.register("population", tools.initRepeat, list, toolbox.particle)
#     toolbox.register("update", updateParticle, c1=pso_params['c1'], c2=pso_params['c2'])
#     toolbox.register("convert", convertQuantum, dist="nuvd")
#     toolbox.register("evaluate", evaluateParticleCost,
#                                  failure_rates = failure_rates,
#                                  service_rates = service_rates,
#                                  holding_costs = holding_costs,
#                                  penalty_cost = penalty_cost,
#                                  machine_cost = machine_cost,
#                                  skill_cost = skill_cost,
#                                  runTheSimulation = runTheSimulation)
#
#
#
#
#     stats = tools.Statistics(lambda ind: ind.fitness.values)
#     stats.register("avg", np.mean)
#     stats.register("std", np.std)
#     stats.register("min", np.min)
#     stats.register("max", np.max)
#
#     logbook = tools.Logbook()
#     logbook.header = ["gen", "evals"] + stats.fields
#
#     GEN = pso_params['generation']
#     best = None
#     # pop = toolbox.population(n=pso_params['population'])
#     pop = [toolbox.swarm(n=pso_params['population']) for _ in range(pso_params['swarms'])]
#
#
#     # Evaluate each particle
#     for swarm in pop:
#         for part in swarm:
#             part.fitness.values = toolbox.evaluate(part)
#
#             # Update swarm's attractors personal best and global best
#             if not part.best or part.fitness > part.bestfit:
#                 part.best = toolbox.clone(part[:])          # Get the position
#                 part.bestfit.values = part.fitness.values   # Get the fitness
#             if not swarm.best or part.fitness > swarm.bestfit:
#                 swarm.best = toolbox.clone(part[:])         # Get the position
#                 swarm.bestfit.values = part.fitness.values  # Get the fitness
#
#
#
#     record = stats.compile(itertools.chain(*pop))
#     logbook.record(gen=0, evals=mpb.nevals, nswarm=len(pop), error=mpb.currentError(), offline_error=mpb.offlineError(), **record)
#
#
#     for g in range(GEN+1):
#         for part in pop:
#             # part.fitness.values = toolbox.evaluate(part)
#             if part.best is None or part.best.fitness < part.fitness:
#                 part.best = creator.Particle(part)
#                 part.best.fitness.values = part.fitness.values
#             if best is None or best.fitness < part.fitness:
#                 best = creator.Particle(part)
#                 best.fitness.values = part.fitness.values
#
#
#         nevals = 0
#         for part in pop:
#             inertia = pso_params['inertia_max'] - g*(pso_params['inertia_max'] - pso_params['inertia_min'])/GEN
#             if toolbox.update(part, best, intertia=inertia):
#                 part.fitness.values = toolbox.evaluate(part)
#                 nevals += 1
#             # toolbox.update(part, best, intertia=inertia)
#             # part.fitness.values = toolbox.evaluate(part)
#
#         # Gather all the fitnesses in one list and print the stats
#         logbook.record(gen=g, evals=len(pop), nevals=nevals, **stats.compile(pop))
#         if verbose:
#             print( best.fitness.values, logbook.stream)
#
#
#     best_assignment = np.array(best).reshape(-1, len(failure_rates))
#
#     totalCost, S, EBO, _, utilizationRatesAverage, skillServerAssignmentStatisitcsAverage =\
#         simulation_optimization_run(failure_rates, service_rates, holding_costs, penalty_cost, best_assignment, runTheSimulation=runTheSimulation)
#     totalCost += skill_cost*np.sum(best_assignment)
#     totalCost +=  machine_cost*sum((np.sum(best_assignment,axis=1)>0))
#
#     np.seterr(**old_settings)
#
#
#     return best_assignment, totalCost, S, EBO, logbook, utilizationRatesAverage, skillServerAssignmentStatisitcsAverage
#
#
#

def test1():

    initDatabase("..//db_simulations_2019",createNewDatabase=False)

    n_servers = 2
    m_skills = 4
    failureRates = np.array([0.21515405370607182, 0.9848459462939283, 0.21415405370607182, 0.6858459462939283])
    serviceRates = np.array([1.0, 1.0, 1.0, 1.0])
    # failureRates = np.array([6]*m_skills, dtype=float)
    # serviceRates = np.array([13]*m_skills, dtype=float)

    n_servers = 6
    m_skills = 3
    failureRates = np.array([0.20073, 1.05844, 2.28659])
    serviceRates = np.array([0.19808, 0.28975, 3.11675])

    holdingCosts = np.ones(m_skills)*1
    penalty = 100.0

    machineCost = 1000
    skillCost = 100
    pso_params = {'generation': 100, 'population':25, 'c1': 1.0, 'c2': 1.0, 'inertia_max': 1.0, 'inertia_min': 10.0, 'speedRange': 0.0, 'swarms':2}

    # print optimizeSkillAssignmentMultiPSO(n_servers, failureRates, serviceRates, holdingCosts, penalty, machineCost, skillCost,
    #                               pso_params=pso_params, seed=123, verbose=True, runTheSimulation=False)

    print( optimizeSkillAssignmentPSO(n_servers, failureRates, serviceRates, holdingCosts, penalty, machineCost, skillCost, caseID = 'TestCase',
                                  pso_params=pso_params, seed=123, verbose=False, showProgressBar=True, runTheSimulation=False))


if __name__ == "__main__":
    # import sys
    # sys.path.insert(0, '/Users/andrei/Documents/Research/Research Papers/Skill Assignment/PythonCodes/')
    # from simulation_codes import simulation_batch_run
    # from simulation_codes import generateSkillsMultiServer, checkSkillAssignment

    start = time.time()
    test1()
    print( "runtime:", time.time() - start)
    pass
