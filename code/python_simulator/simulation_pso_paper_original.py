# from __future__ import division

import bisect
import multiprocessing as mp
import itertools
import time
from collections import deque

import numpy as np
import numpy.random as rnd

olderr = np.seterr(divide='ignore')

# import skill_set_generator as sk_gen
# import itertools

curTime = 0
eventList = deque([])
servers = []
# freeServers = []
waitingRequests = dict()

# waiting time statistics
nArrivals = dict()
waitingTimeMean = dict()
waitingTimeMeanSquare = dict()
waitingTimeVariance = dict()
waitingTimeHistogram = dict()
waitingTimesDataset = dict()
histogramBuckets = None

sojournTimeMean = dict()
sojournTimeMeanSquare = dict()
sojournTimeVariance = dict()
sojournTimeHistogram = dict()
sojournTimesDataset = dict()

numberInSystemDistribution = dict()
lastNrInSystemUpdate = dict()
currentNumberInSystem = dict()

numberInSystemDistributionAverage = dict()

utilizationRates = None
skillServerAssignmentStatisitcs = None

collectWaitingTimeStatistics = False

warmingUpTime = 0

def resetGlobals():
    """
        Resets global parameters
        :rtype: None
    """
    global curTime
    global eventList
    global servers
    # global freeServers
    global waitingRequests

    # waiting time statistics
    global nArrivals
    global waitingTimeMean
    global waitingTimeMeanSquare
    global waitingTimeVariance
    global waitingTimeHistogram
    global waitingTimesDataset

    global sojournTimeMean
    global sojournTimeMeanSquare
    global sojournTimeVariance
    global sojournTimeHistogram
    global sojournTimesDataset

    global numberInSystemDistribution
    global lastNrInSystemUpdate
    global currentNumberInSystem


    curTime = 0
    eventList = [] # deque([])
    servers = []
    # freeServers = []
    waitingRequests = dict()


    # waiting time statistics
    nArrivals = dict()
    waitingTimeMean = dict()
    waitingTimeMeanSquare = dict()
    waitingTimeVariance = dict()
    waitingTimeHistogram = dict()
    waitingTimesDataset = dict()

    sojournTimeMean = dict()
    sojournTimeMeanSquare = dict()
    sojournTimeVariance = dict()
    sojournTimeHistogram = dict()
    sojournTimesDataset = dict()

    numberInSystemDistribution = dict()
    lastNrInSystemUpdate = dict()
    currentNumberInSystem = dict()


class Request:
    def __init__(self,
                 requestedAt = 0,
                 # priority = 100,
                 # expServiceRate = 0,
                 requiredSkill = -1):
        self.requestedAt = requestedAt
        self.startWaitingAt = requestedAt  # will be used for postponed items
        # self.priority = priority
        self.requiredSkill = requiredSkill
        # self.expServiceRate = expServiceRate
        self.waitingTime = 0.0
        self.totalTime = 0.0
        self.serviceTime = 0.0
        self.startProcessingAt = np.inf

        Server.updateQueuesStatistic(curTime, requiredSkill, 1)

        if not self.requiredSkill in nArrivals: nArrivals[self.requiredSkill] = 0
        nArrivals[self.requiredSkill] += 1


    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Request(arr:"+self.requestedAt.__str__()+", sk:" + self.requiredSkill.__str__()+", RT:" + self.serviceTime.__str__()\
               +", StAt:" + self.startProcessingAt.__str__()+")"

    def __lt__(self, other):
        return self.requestedAt < other.requestedAt
    # @profile
    def getPossibleServer(self, servers):

        possibleServers  = filter(lambda t: t.requestToServe == None or t.serverSkillScore[self.requiredSkill]< \
                                            t.serverSkillScore[t.requestToServe.requiredSkill], servers)
        possibleServers1 = filter(lambda t: t.requestToServe == None or t.serverSkillScore[self.requiredSkill]< \
                                            t.serverSkillScore[t.requestToServe.requiredSkill], servers)
        # two filters are needed here

        # if any(possibleServers): self.bestFreeServer = min(possibleServers)
        # self.bestFreeServer = min(possibleServers, key=lambda t: t.serverSkillScore[self.requiredSkill] * (t.requestToServe != None)) \
        self.bestFreeServer = min(possibleServers, key=lambda t: t.serverSkillScore[self.requiredSkill] if t.requestToServe else 0) \
                    if any(possibleServers1) else None
        # self.bestFreeServer = min(filter(lambda t: t.serverSkillScore[self.requiredSkill] < np.inf, servers)) if any(servers) else None
        # if len(servers):
        #     possibleServers = list(filter(lambda t: t.serverSkillScore[self.requiredSkill]< float('inf'), servers))
        #     if len(possibleServers):
        #         self.bestFreeServer = min(possibleServers, key=lambda t: t.serverSkillScore[self.requiredSkill])

        # if self.bestFreeServer:
        #     servers.remove(self.bestFreeServer)

    def getServerScore(self, server):
        return server.serverSkillScore[self.requiredSkill]

    def updateRequestStatistics(self):
        global nArrivals
        global waitingTimeMean
        global waitingTimeMeanSquare
        global waitingTimeVariance
        global waitingTimeHistogram
        global waitingTimesDataset

        global sojournTimeMean
        global sojournTimeMeanSquare
        global sojournTimeVariance
        global sojournTimeHistogram
        global sojournTimesDataset



        if not self.requiredSkill in waitingTimeMean:
            waitingTimeMean[self.requiredSkill] = 0.0
            waitingTimeMeanSquare[self.requiredSkill] = 0.0
            waitingTimeVariance[self.requiredSkill] = 0.0
            # waitingTimeHistogram[self.requiredSkill] = dict()
            waitingTimesDataset[self.requiredSkill] = []

            sojournTimeMean[self.requiredSkill] = 0.0
            sojournTimeMeanSquare[self.requiredSkill] = 0.0
            sojournTimeVariance[self.requiredSkill] = 0.0
            sojournTimeHistogram[self.requiredSkill] = dict()
            sojournTimesDataset[self.requiredSkill] = []

        waitingTimeMean[self.requiredSkill] = waitingTimeMean[self.requiredSkill]*(nArrivals[self.requiredSkill] - 1) \
                /nArrivals[self.requiredSkill] + self.waitingTime/nArrivals[self.requiredSkill]
        waitingTimeMeanSquare[self.requiredSkill] = waitingTimeMeanSquare[self.requiredSkill]*(nArrivals[self.requiredSkill] - 1) \
                /nArrivals[self.requiredSkill] + self.waitingTime**2/nArrivals[self.requiredSkill]
        waitingTimeVariance[self.requiredSkill] = (waitingTimeMeanSquare[self.requiredSkill] - waitingTimeMean[self.requiredSkill]**2) \
                *nArrivals[self.requiredSkill]/max(1.0,(float)(nArrivals[self.requiredSkill]-1)) #max here plays role of DIV0 catcher

        sojournTimeMean[self.requiredSkill] = sojournTimeMean[self.requiredSkill]*(nArrivals[self.requiredSkill] - 1) \
                /nArrivals[self.requiredSkill] + self.totalTime/nArrivals[self.requiredSkill]
        sojournTimeMeanSquare[self.requiredSkill] = sojournTimeMeanSquare[self.requiredSkill]*(nArrivals[self.requiredSkill] - 1) \
                /nArrivals[self.requiredSkill] + self.totalTime**2/nArrivals[self.requiredSkill]
        sojournTimeVariance[self.requiredSkill] = (sojournTimeMeanSquare[self.requiredSkill] - sojournTimeMean[self.requiredSkill]**2) \
                *nArrivals[self.requiredSkill]/max(1.0,(float)(nArrivals[self.requiredSkill]-1)) #max here plays role of DIV0 catcher

        # total statistics collection is switched off for performance reasons
        # waitingTimesDataset[self.requiredSkill] = waitingTimesDataset[self.requiredSkill] + [self.waitingTime]
        # sojournTimesDataset[self.requiredSkill] = sojournTimesDataset[self.requiredSkill] + [self.totalTime]


        # tmIndex = (int)(self.waitingTime)
        # if not tmIndex in waitingTimeHistogram[self.requiredSkill]:  waitingTimeHistogram[self.requiredSkill][tmIndex]=0
        # waitingTimeHistogram[self.requiredSkill][tmIndex] +=1

        waitingTimeHistogram[self.requiredSkill] += (self.waitingTime <= histogramBuckets)


    @staticmethod
    def scheduleNextRequestArrival(interArrivalTime=0):
        Event.addEvent(interArrivalTime, "RequestArrival", None)


class Server:
    def __init__(self,
                 serverID=0,
                 requestProcessingRate =[],
                 requestProcessingCost =[],
                 serverSkillScore = [],
                 requestToServe = None):
        self.serverID=serverID
        self.requestProcessingRate = requestProcessingRate
        self.requestProcessingCost = requestProcessingCost
        self.requestToServe = requestToServe
        self.serverSkillScore = serverSkillScore
        self.endOfServiceEvent = None

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        reqToServe = "none"
        if(self.requestToServe):
            reqToServe = self.requestToServe.__str__()
        return "Server("+self.serverID.__str__() +"," +self.requestProcessingRate.__str__() +"," + self.requestProcessingCost.__str__()  +"," + reqToServe+")"


    # @profile
    def pickNextRequest(self):
        requestToServe=[]

        bestScore = np.inf #float('inf')
        for q in waitingRequests:
            if len(waitingRequests[q])  and  self.serverSkillScore[q] != np.inf and  self.serverSkillScore[q] <= bestScore:
                if self.serverSkillScore[q] < bestScore:
                    bestScore = self.serverSkillScore[q]
                    requestToServe = [waitingRequests[q][0]]
                elif self.serverSkillScore[q] == bestScore:
                    requestToServe.append(waitingRequests[q][0])

        return min(requestToServe) if len(requestToServe) else None

    # def pickNextRequestV2(self):
    #     best_qs = None
    #
    #     bestScore = np.inf #float('inf')
    #     for q in waitingRequests:
    #         if len(waitingRequests[q])  and  self.serverSkillScore[q] != np.inf and  self.serverSkillScore[q] <= bestScore:
    #             if self.serverSkillScore[q] < bestScore:
    #                 bestScore = self.serverSkillScore[q]
    #                 best_qs = [q]
    #             elif self.serverSkillScore[q] == bestScore:
    #                 best_qs.append(q)
    #
    #     requestToServe = None
    #     if best_qs:
    #         # postponed requests
    #         postponed =[]
    #         for q in best_qs:
    #             postponed += [r for r in waitingRequests[q] if r.serviceTime]
    #
    #         if len(postponed):
    #             requestToServe = postponed[int(rnd.uniform()*len(postponed))]
    #         else:
    #             requestToServe = max(waitingRequests[q][0] for q in best_qs)
    #
    #     return requestToServe

    # pickNextRequest = pickNextRequestV1
    # pickNextRequest = pickNextRequestV2

    # @profile
    def startRequestProcessing(self, req, serviceTime):
        self.postponeServedRequest()

        self.endOfServiceEvent = Event.addEvent(serviceTime, "ServerEOS", self)
        try:
            req = waitingRequests[req.requiredSkill].popleft()
        except:
            pass  #do nothing if customer was not added
        self.requestToServe=req
        req.waitingTime += curTime - req.startWaitingAt
        req.startProcessingAt = curTime
        req.serviceTime = serviceTime

    # @profile
    def postponeServedRequest(self):
        if self.requestToServe:
            req = self.requestToServe
            req.serviceTime -= curTime - req.startProcessingAt
            bisect.insort(waitingRequests[req.requiredSkill], req)
            eventList.remove(self.endOfServiceEvent)

    # @profile
    def requestServed(self):
        global curTime
        global utilizationRates
        global skillServerAssignmentStatisitcs

        self.requestToServe.totalTime = curTime  - self.requestToServe.requestedAt

        if collectWaitingTimeStatistics:
            self.requestToServe.updateRequestStatistics()


        skill = self.requestToServe.requiredSkill

        utilizationRates[self.serverID, self.requestToServe.requiredSkill] += curTime - self.requestToServe.startProcessingAt
        skillServerAssignmentStatisitcs[self.serverID, self.requestToServe.requiredSkill] += 1

        Server.updateQueuesStatistic(curTime, skill, -1)

        self.requestToServe = None


    @staticmethod
    # @profile
    def updateQueuesStatistic(curTime, skill, queueChange):
        global numberInSystemDistribution
        global lastNrInSystemUpdate
        global currentNumberInSystem

        if curTime > warmingUpTime:
            if not currentNumberInSystem[skill] in numberInSystemDistribution[skill]:
                numberInSystemDistribution[skill][currentNumberInSystem[skill]] = (curTime - lastNrInSystemUpdate[skill])
            else:
                numberInSystemDistribution[skill][currentNumberInSystem[skill]] += (curTime - lastNrInSystemUpdate[skill])

        currentNumberInSystem[skill] += queueChange
        lastNrInSystemUpdate[skill] = curTime





class Event:
    eventTime=0
    eventType="NA"
    #should take only values
    #    "NA" - not defined
    #    "RequestArrival" - next customer arrival
    #    "ServerEOS" - taxi end of service

    def __init__(self, eventTime, eventType, eventAppliedTo):
        self.eventTime=eventTime
        self.eventType=eventType
        self.eventAppliedTo=eventAppliedTo

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        appliedTo = "none"
        if(self.eventAppliedTo):
            appliedTo = self.eventAppliedTo.__str__()
        return "Event(" + self.eventType.__str__()  +", "+ self.eventTime.__str__() +", " +  appliedTo +")"

    def __lt__(self, other):
        return self.eventTime > other.eventTime  # for reverse ordering, it allows us to use .pop() instead of .pop(0)


    @staticmethod
    # @profile
    def addEvent(timeNeeded=0, eventType=0, eventAppliedTo=None):
        global curTime
        event = Event(curTime + timeNeeded, eventType, eventAppliedTo)
        bisect.insort(eventList, event)
        return event

    @staticmethod
    def processGlobalStatistics():
        #TODO
        pass



# @profile
def simulation(  requestArrivalRate = 5.0,
                 skillDistribution = [],
                 skillServerRates = [],
                 skillServerCosts = [],
                 histogramBucketsInput = [],
                 expectedTotalNumberRequests = 10000,
                 numberWarmingUpRequests = 1000,
                 seed = 10,
                 collectWaitingTimeStatisticsInput = False
               ):

    global warmingUpTime

    stopTime = expectedTotalNumberRequests/requestArrivalRate
    warmingUpTime = numberWarmingUpRequests/requestArrivalRate

    rnd.seed(seed)
    resetGlobals()

    global histogramBuckets
    global utilizationRates
    global skillServerAssignmentStatisitcs
    global collectWaitingTimeStatistics

    collectWaitingTimeStatistics = collectWaitingTimeStatisticsInput

    histogramBuckets = np.array(histogramBucketsInput)

    utilizationRates = np.zeros(skillServerRates.shape)
    skillServerAssignmentStatisitcs = np.zeros(skillServerRates.shape)

    numberOfServer = len(skillServerRates)

    serverSkillScore = dict()
    for sr in range(numberOfServer):
        serverSkillScore[sr] = dict()
        for sk in range(len(skillDistribution)):
            serverSkillScore[sr][sk] = np.round(1.0/skillServerRates[sr, sk] * skillServerCosts[sr,sk],6) if (skillServerRates[sr,sk]) else np.inf #float('inf')

    servers = [ Server(serverID=i,
                           requestProcessingRate = skillServerRates[i],
                           requestProcessingCost = skillServerCosts[i],
                           serverSkillScore = serverSkillScore[i],
                          ) for i in range(numberOfServer)]

    # servers = freeServers[:]
    for sk in range(len(skillDistribution)):
        waitingRequests[sk] = deque([])
        waitingTimeHistogram[sk] = np.zeros(len(histogramBuckets))



    global curTime
    curTime=0.0

    global numberInSystemDistribution
    global lastNrInSystemUpdate
    global currentNumberInSystem
    global numberInSystemDistributionAverage

    for sk in range(len(skillDistribution)):
        numberInSystemDistribution[sk] = dict()
        currentNumberInSystem[sk] = 0
        lastNrInSystemUpdate[sk] = 0.0
    expectedInterArrivalTime = 1.0/requestArrivalRate
    expectedSkillServerTimes = 1.0/np.array(skillServerRates)

    Request.scheduleNextRequestArrival(rnd.exponential(expectedInterArrivalTime))

    while curTime < stopTime:# and sum(nArrivals.values()) < 20:
        #get nextEvent

        nextEvent = eventList.pop() # pop(0); #removes event from the list

        curTime = nextEvent.eventTime

        if (nextEvent.eventType=="RequestArrival"): #customer arrival
            Request.scheduleNextRequestArrival(rnd.exponential(expectedInterArrivalTime))
            skillProb = rnd.uniform(0, 1)

            # skill = sum((p<=skillProb) for p in skillDistribution)
            # skill = np.sum(skillDistribution<=skillProb)
            skill = 0; skillBools = skillDistribution<=skillProb
            for x in skillBools:
                if x:
                    skill += 1
                else:
                    break
            # this is a faster version of the skill selector

            #customerType is generated on the arriving customer
            #this doesn't contradict the model if we assume Poisson arrival processes
            newRequest = Request(requestedAt =curTime,
                                 requiredSkill=skill
                                 )

            newRequest.getPossibleServer(servers)
            waitingRequests[newRequest.requiredSkill].append(newRequest)
            if newRequest.bestFreeServer:
                serviceTime = rnd.exponential(expectedSkillServerTimes[newRequest.bestFreeServer.serverID,newRequest.requiredSkill])
                newRequest.bestFreeServer.startRequestProcessing(newRequest, serviceTime)


        if (nextEvent.eventType=="ServerEOS"): #end of service
            appliedTo = nextEvent.eventAppliedTo
            appliedTo.requestServed()
            requestToServe = appliedTo.pickNextRequest()

            if requestToServe:
                serviceTime = rnd.exponential(expectedSkillServerTimes[appliedTo.serverID, requestToServe.requiredSkill])
                appliedTo.startRequestProcessing(requestToServe, serviceTime)
            # else:
            #     servers.append(appliedTo)


        Event.processGlobalStatistics()

        # print(eventList)

    for sk in numberInSystemDistribution:
        if not sk in numberInSystemDistributionAverage:
            numberInSystemDistributionAverage[sk] = dict()
        for n in numberInSystemDistribution[sk]:
            numberInSystemDistribution[sk][n] /= (curTime - warmingUpTime)

            if not n in numberInSystemDistributionAverage[sk]:
                numberInSystemDistributionAverage[sk][n] = numberInSystemDistribution[sk][n]
            else:
                numberInSystemDistributionAverage[sk][n] += numberInSystemDistribution[sk][n]


    for pr in nArrivals:
        waitingTimeHistogram[pr] /= (float)(nArrivals[pr])

    utilizationRates /= curTime

    for sk in nArrivals:
        skillServerAssignmentStatisitcs[:,sk] /= nArrivals[sk]

    return numberInSystemDistribution, waitingTimeHistogram, utilizationRates, skillServerAssignmentStatisitcs

def simulation_batch_run(failureRates,
                         serviceRates,
                         skillServerAssignment,
                         priorities = None,
                         histogramBucketsInput = [0.0],
                         expectedTotalNumberRequests = 10000,
                         numberWarmingUpRequests = 1000,
                         replications=2,
                         maxQueue=10,
                         nCores=None,
                         startSeed = 10):

    pool = mp.Pool(processes=nCores)

    requestArrivalRateInput = np.sum(failureRates)
    skillDistributionInput = np.cumsum(failureRates/requestArrivalRateInput)
    skillServerCosts = np.ones(skillServerAssignment.shape)
    skillServerRates = np.transpose(np.zeros(skillServerAssignment.shape))
    for i in range(serviceRates.shape[0]): skillServerRates[i,:] = skillServerAssignment[:,i] * serviceRates[i]
    skillServerRates = np.transpose(skillServerRates)

    numberInSystemDistributionAverage = np.zeros((len(failureRates), maxQueue))
    waitingTimeHistogramAverage = np.zeros((len(failureRates), len(histogramBucketsInput)))
    utilizationRatesAverage = np.zeros(skillServerRates.shape)
    skillServerAssignmentStatisitcsAverage = np.zeros(skillServerRates.shape)
    skillServerCosts[:] = skillServerRates
    if isinstance(priorities, np.ndarray):
        skillServerCosts *= priorities


    try:
        results = [pool.apply_async(simulation,
                     args=(requestArrivalRateInput,
                        skillDistributionInput,  #The distribution should be in cummulative form
                        skillServerRates, ##replenishment times can be different for different priority classes
                        skillServerCosts,
                        histogramBucketsInput,
                        expectedTotalNumberRequests,
                        numberWarmingUpRequests,
                        startSeed+20*i,
                        ) )
            for i in range(replications)]

        for p in results:
            numberInSystemDistribution = p.get()[0]
            waitingTimeHistogram = p.get()[1]
            utilizationRates = p.get()[2]
            skillServerAssignmentStatisitcs = p.get()[3]

            for sk in numberInSystemDistribution:
                for n in numberInSystemDistribution[sk]:
                    if(n<maxQueue):
                        numberInSystemDistributionAverage[sk,n] += numberInSystemDistribution[sk][n]
                waitingTimeHistogramAverage[sk,:] += waitingTimeHistogram[sk]
            utilizationRatesAverage += utilizationRates
            skillServerAssignmentStatisitcsAverage += skillServerAssignmentStatisitcs


        #print "Runs are executed:", len(results)
        pool.close()
    except KeyboardInterrupt:
        print('got ^C while pool mapping, terminating the pool')
        pool.terminate()
        print('pool is terminated')
    except Exception as e:
        print('got exception: %r, terminating the pool' % (e,))
        pool.terminate()
        print('pool is terminated')
    finally:
        #print 'joining pool processes'
        pool.join()
        #print 'join complete'
    #print 'the end'

    # print numberInSystemDistributionAverage
    numberInSystemDistributionAverage /= replications
    waitingTimeHistogramAverage /= replications
    utilizationRatesAverage /= replications
    skillServerAssignmentStatisitcsAverage /= replications



    return numberInSystemDistributionAverage, waitingTimeHistogramAverage, utilizationRatesAverage, skillServerAssignmentStatisitcsAverage


def run_time_test():
    np.set_printoptions(linewidth=50000)
    startTime = time.time()

    expectedTotalNumberRequests = 50000
    replications = 70
    maxQueue = 500
    nCores = None  # maximum number of cores will be used

    n_servers = 2
    m_skills = 4
    failureRates = np.array([0.21515405370607182, 0.9848459462939283, 0.21515405370607182, 0.9848459462939283])
    serviceRates = np.array([1.0, 1.0, 1.0, 1.0])
    skillServerAssignment = np.array([[0,1,0,1],[1,0,1,0]])


    numberInSystemDistributionAverage, waitingTimeHistogramAverage, utilizationRatesAverage, skillServerAssignmentStatisitcsAverage \
       = simulation_batch_run(failureRates, serviceRates, skillServerAssignment,
                                 histogramBucketsInput = np.array(range(10)),
                                 expectedTotalNumberRequests = expectedTotalNumberRequests,
                                 numberWarmingUpRequests = 1000,
                                 replications=replications,
                                 maxQueue=maxQueue,
                                 nCores=nCores)
    print("n_servers", n_servers)
    print("m_skills", m_skills)
    print("stopTime", expectedTotalNumberRequests)
    print("replications", replications)

    print("Runtime:", time.time() - startTime)
    en = np.sum(np.array(range(maxQueue))*numberInSystemDistributionAverage, axis=1)
    varn = np.sum(np.array(range(maxQueue))*np.array(range(maxQueue))*numberInSystemDistributionAverage, axis=1) - en**2
    print(en, varn)


    pass

def run_time_test_single_item():
    np.set_printoptions(linewidth=50000)
    startTime = time.time()

    expectedTotalNumberRequests = 500000
    replications = 20
    maxQueue = 500
    nCores = None  # maximum number of cores will be used

    n_servers = 2
    m_skills = 4
    failureRates = np.array([0.21515405370607182, 0.5848459462939283, 0.21515405370607182, 0.5848459462939283])
    serviceRates = np.array([0.5, 2.0, 1.0/2, 2.0])
    skillServerAssignment = np.array([[1,1,1,1],[1,1,1,1]])

    requestArrivalRateInput = np.sum(failureRates)
    skillDistributionInput = np.cumsum(failureRates/requestArrivalRateInput)
    skillServerCosts = np.ones(skillServerAssignment.shape)
    skillServerRates = np.transpose(np.zeros(skillServerAssignment.shape))
    for i in range(serviceRates.shape[0]): skillServerRates[i,:] = skillServerAssignment[:,i] * serviceRates[i]
    skillServerRates = np.transpose(skillServerRates)
    skillServerCosts = np.array([[1, 1, 2, 2],[1, 1, 2, 2]])*skillServerRates



    numberInSystemDistribution, waitingTimeHistogram, utilizationRates, skillServerAssignmentStatisitcs \
        = simulation(  requestArrivalRate = requestArrivalRateInput,
                 skillDistribution = skillDistributionInput,
                 skillServerRates = skillServerRates,
                 skillServerCosts = skillServerCosts,
                 histogramBucketsInput= np.array(range(10)),
                 expectedTotalNumberRequests = expectedTotalNumberRequests,
                 numberWarmingUpRequests = 0,
                 seed = 25,
               )


    print("n_servers", n_servers)
    print("m_skills", m_skills)
    print("stopTime", expectedTotalNumberRequests)

    print("Runtime:", time.time() - startTime)

    pass


def tune_runtime_batchsize():
    import itertools
    import QueueingModels as qm
    np.set_printoptions(linewidth=50000)
    startTime = time.time()

    expectedTotalNumberRequestsRange = [100000] #[50000, 100000]
    replicationsRange = [20, 25, 30, 35 ]#50,60,70,80,90, 100]
    maxQueue = 50
    nCores = None  # maximum number of cores will be used

    failureRates = np.array([0.45, 0.45, 0.45, 0.35])
    serviceRates = np.array([1.0, 1.0, 1.0, 1.0])
    # skillServerAssignment = np.array([[0, 1], [0, 1], [1, 0], [1, 0]])
    skillServerAssignment = np.array([[1,1,1,1], [1,1,1,1]])

    # failureRates = np.array([ 0.30558721, 0.48466022, 0.20975257])
    # serviceRates = np.array([ 3.49403387, 0.62282548, 1.09848092])
    # skillServerAssignment = np.array([[1, 1, 1], [1, 1, 1]])

    histogramBucketsInput = np.array(range(10))

    exactDistribution = np.zeros((4, maxQueue))
    ''' build on the principle that in the shared single server queue
        the marginal distribution can be computed as M/M/1 queue with
        new_rho_i = rho_i/(1-rho+rho_i)
    '''
    # these two failures share one queue
    exactDistribution[:2,:] = qm.ComputeMarginalDistributions(failureRates[:2], serviceRates[:2], nServers=1, maxQueueLength = maxQueue)
    # rho = sum(failureRates[:2])
    # rho_i = failureRates[0]/(1 - rho + failureRates[0])
    # exactDistribution[0,:] = MMC(rho_i, 1.0, 1, self.maxQueue)
    #
    # rho_i = failureRates[1]/(1 - rho + failureRates[1])
    # exactDistribution[1,:] = MMC(rho_i, 1.0, 1, self.maxQueue)



    # these two failures share the other queue
    exactDistribution[2:,:] = qm.ComputeMarginalDistributions(failureRates[2:], serviceRates[2:], nServers=1, maxQueueLength = maxQueue)
    # rho = sum(failureRates[2:])
    # rho_i = failureRates[2]/(1 - rho + failureRates[2])
    # exactDistribution[2,:] =MMC(rho_i, 1.0, 1, self.maxQueue)
    # rho_i = failureRates[3]/(1 - rho + failureRates[3])
    # exactDistribution[3,:] = MMC(rho_i, 1.0, 1, self.maxQueue)

    bestRunTime = np.inf #float('inf')
    bestParams = ()
    for (expectedTotalNumberRequests, replications) in itertools.product(expectedTotalNumberRequestsRange, replicationsRange):

        for _ in range(1):
            start = time.time()
            numberInSystemDistributionAverage, waitingTimeHistogramAverage, utilizationRatesAverage, skillServerAssignmentStatisitcsAverage \
               = simulation_batch_run(failureRates, serviceRates, skillServerAssignment,
                                         histogramBucketsInput = histogramBucketsInput,
                                         expectedTotalNumberRequests = expectedTotalNumberRequests,
                                         numberWarmingUpRequests = 1000,
                                         replications = replications,
                                         maxQueue= maxQueue)

            runtime = time.time() - start
            diffs = np.max(numberInSystemDistributionAverage - exactDistribution, axis=1)
            comparable = np.sum(diffs < 1e-03) == len(diffs)
            # comparable = np.allclose(numberInSystemDistributionAverage, exactDistribution, atol=1e-03)

            print(expectedTotalNumberRequests, replications, runtime, np.max(numberInSystemDistributionAverage - exactDistribution, axis=1), comparable)
            if comparable and runtime < bestRunTime:
                bestParams = (expectedTotalNumberRequests, replications)
                bestRunTime = runtime
        # break

    print("Best Runtime:", bestRunTime, " | BestParams:", bestParams)
    pass



if __name__ == '__main__':
    # run_time_test()
    run_time_test_single_item()
    # tune_runtime_batchsize()
    pass


