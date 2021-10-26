#!/usr/bin/env python
import random, sys

from engine.const import Const
import graderUtil
import util
import collections
import copy

grader = graderUtil.Grader()
submission = grader.load('submission')

try:
    import solution
    grader.addHiddenPart = grader.addBasicPart
    SEED = solution.SEED
    solution_exist = True
except ModuleNotFoundError:
    SEED = 42
    solution_exist = False

############################################################
# Problem 1: Warmup

def test_1a_1():
    kargs = dict(delta=0.5, epsilon=0.7, eta=0.3, c2=1, d2=0)
    pred = submission.get_conditional_prob1(**kargs)
    grader.requireIsEqual(0.3, pred, 0.00001)

grader.addBasicPart('1a-1-basic', test_1a_1, 0, maxSeconds=1, description="Basic test for problem-1a")

def test_1a_2():
    random.seed(SEED)
    def get_cmp_gen():
        for _ in range(20):
            kargs = dict(delta=random.uniform(0.4, 0.6),
                         epsilon=random.uniform(0.4, 0.6),
                         eta=random.uniform(0.4, 0.6),
                         c2=random.randint(0, 1),
                         d2=random.randint(0, 1))
            pred = submission.get_conditional_prob1(**kargs)
            if solution_exist:
                answer = solution.get_conditional_prob1(**kargs)
                yield grader.requireIsEqual(answer, pred, 0.00001)
    all(get_cmp_gen())

grader.addHiddenPart('1a-2-hidden', test_1a_2, 4, maxSeconds=1, description="Hidden test for problem-1a")


def test_1b_1():
    kargs = dict(delta=0.5, epsilon=0.7, eta=0.3, c2=1, d2=0, d3=1)
    pred = submission.get_conditional_prob2(**kargs)
    grader.requireIsEqual(0.23684210526315788, pred, 0.00001)

grader.addBasicPart('1b-1-basic', test_1b_1, 0, maxSeconds=1, description="Basic test for problem-1b")

def test_1b_2():
    random.seed(SEED)
    def get_cmp_gen():
        for _ in range(20):
            kargs = dict(delta=random.uniform(0.4, 0.6),
                         epsilon=random.uniform(0.4, 0.6),
                         eta=random.uniform(0.4, 0.6),
                         c2=random.randint(0, 1),
                         d2=random.randint(0, 1),
                         d3=random.randint(0, 1))
            pred = submission.get_conditional_prob2(**kargs)
            if solution_exist:
                answer = solution.get_conditional_prob2(**kargs)
                yield grader.requireIsEqual(answer, pred, 0.00001)
    all(get_cmp_gen())

grader.addHiddenPart('1b-2-hidden', test_1b_2, 4, maxSeconds=1, description="Hidden test for problem-1b")

def test_1c_1():
    random.seed(SEED)

    epsilon = submission.get_epsilon()
    assert 0 <= epsilon <= 1

    def get_cmp_gen():
        for _ in range(20):
            kargs1 = dict(delta=random.uniform(0.4, 0.6),
                          epsilon=epsilon,
                          eta=random.uniform(0.4, 0.6),
                          c2=random.randint(0, 1),
                          d2=random.randint(0, 1))
            kargs2 = dict(kargs1)
            kargs2.update(dict(d3=random.randint(0, 1)))

            pred1 = submission.get_conditional_prob1(**kargs1)
            pred2 = submission.get_conditional_prob2(**kargs2)

            yield grader.requireIsEqual(pred1, pred2, 0.00001)
    all(get_cmp_gen())

grader.addHiddenPart('1c-1-hidden', test_1c_1, 2, maxSeconds=1, description="Hidden test for problem-1c")

############################################################
# Problem 2: Emission probabilities

def test_2a():
    ei = submission.ExactInference(10, 10)
    ei.skipElapse = True ### ONLY FOR PROBLEM 2
    ei.observe(55, 193, 200)
    grader.requireIsEqual(0.030841805296, ei.belief.getProb(0, 0), 0.00001)
    grader.requireIsEqual(0.00073380582967, ei.belief.getProb(2, 4), 0.00001)
    grader.requireIsEqual(0.0269846478431, ei.belief.getProb(4, 7), 0.00001)
    grader.requireIsEqual(0.0129150762582, ei.belief.getProb(5, 9), 0.00001)

    ei.observe(80, 250, 150)
    grader.requireIsEqual(0.00000261584106271, ei.belief.getProb(0, 0), 0.00001)
    grader.requireIsEqual(0.000924335357194, ei.belief.getProb(2, 4), 0.00001)
    grader.requireIsEqual(0.0295673460685, ei.belief.getProb(4, 7), 0.00001)
    grader.requireIsEqual(0.000102360275238, ei.belief.getProb(5, 9), 0.00001)

grader.addBasicPart('2a-0-basic', test_2a, 2, description="2a basic test for emission probabilities")

def get_point_gen(numRows, numCols, numSamples):
    for _ in range(numSamples):
        yield random.randint(0, numRows - 1), random.randint(0, numCols - 1)

def test_2a_1(): # test whether they put the pdf in the correct order
    random.seed(10)

    oldpdf = util.pdf
    del util.pdf
    def pdf(a, b, c): # be super rude to them! You can't swap a and c now!
      return a + b
    util.pdf = pdf

    observations = [(55, 193, 200), (80, 250, 150)]

    numRows, numCols = 10, 10
    sub = submission.ExactInference(numRows, numCols)
    sub.skipElapse = True ### ONLY FOR PROBLEM 2

    if solution_exist:
        sol = solution.ExactInference(10, 10)
        sol.skipElapse = True ### ONLY FOR PROBLEM 2

    def get_cmp_gen():
        for observation in observations:
            sub.observe(*observation)
            if solution_exist:
               sol.observe(*observation)
               for point in get_point_gen(numRows, numCols, 20):
                   yield grader.requireIsEqual(sol.belief.getProb(*point),
                                               sub.belief.getProb(*point), 0.00001)
    all(get_cmp_gen())

    util.pdf = oldpdf # replace the old pdf

grader.addHiddenPart('2a-1-hidden',test_2a_1, 1, description="2a test ordering of pdf")

def test_2a_2():
    random.seed(10)

    numRows, numCols = 10, 10
    sub = submission.ExactInference(numRows, numCols)
    sub.skipElapse = True ### ONLY FOR PROBLEM 2

    if solution_exist:
        sol = solution.ExactInference(10, 10)
        sol.skipElapse = True ### ONLY FOR PROBLEM 2

    def get_observation_gen():
        N = 50
        p_values = []
        for i in range(N):
            a = int(random.random() * 300)
            b = int(random.random() * 5)
            c = int(random.random() * 300)
            yield a, b, c

    def get_cmp_gen():
        for observation in get_observation_gen():
            sub.observe(*observation)
            if solution_exist:
               sol.observe(*observation)
               for point in get_point_gen(numRows, numCols, 20):
                   yield grader.requireIsEqual(sol.belief.getProb(*point),
                                               sub.belief.getProb(*point), 0.00001)
    all(get_cmp_gen())

grader.addHiddenPart('2a-2-hidden', test_2a_2, 1, description="2a advanced test for emission probabilities")

############################################################
# Problem 3: Transition probabilities

def test_3a():
    ei = submission.ExactInference(30, 13)
    ei.elapseTime()
    grader.requireIsEqual(0.0105778989624, ei.belief.getProb(16, 6), 0.00001)
    grader.requireIsEqual(0.00250560512469, ei.belief.getProb(18, 7), 0.00001)
    grader.requireIsEqual(0.0165024135157, ei.belief.getProb(21, 7), 0.00001)
    grader.requireIsEqual(0.0178755550388, ei.belief.getProb(8, 4), 0.00001)

    ei.elapseTime()
    grader.requireIsEqual(0.0138327373012, ei.belief.getProb(16, 6), 0.00001)
    grader.requireIsEqual(0.00257237608713, ei.belief.getProb(18, 7), 0.00001)
    grader.requireIsEqual(0.0232612833688, ei.belief.getProb(21, 7), 0.00001)
    grader.requireIsEqual(0.0176501876956, ei.belief.getProb(8, 4), 0.00001)

grader.addBasicPart('3a-0-basic', test_3a, 2, description="test correctness of elapseTime()")

def test_3a_1(): # stress test their elapseTime
    random.seed(15)
    numRows, numCols = 30, 30
    sub = submission.ExactInference(numRows, numCols)
    if solution_exist:
        sol = solution.ExactInference(numRows, numCols)

    N1 = 20
    N2 = 400
    p_values = []

    def get_cmp_gen():
        for i in range(N1):
            sub.elapseTime()
            if solution_exist:
                sol.elapseTime()
            for point in get_point_gen(numRows, numCols, N2):
                pred = sub.belief.getProb(*point)
                if solution_exist:
                    answer = sol.belief.getProb(*point)
                    yield grader.requireIsEqual(answer, pred, 0.00001)
    all(get_cmp_gen())

grader.addHiddenPart('3a-1-hidden',test_3a_1, 1, description="advanced test for transition probabilities, strict time limit", maxSeconds=5)

def test_3a_2(): # let's test them together! Very important
    random.seed(20)
    numRows, numCols = 30, 30
    sub = submission.ExactInference(numRows, numCols)
    if solution_exist:
        sol = solution.ExactInference(numRows, numCols)

    def get_observation_gen():
        N1 = 20
        for i in range(N1):
            a = int(random.random() * 5 * numRows)
            b = int(random.random() * 5)
            c = int(random.random() * 5 * numRows)
            yield a, b, c

    N2 = 400
    def get_cmp_gen():
        for observation in get_observation_gen():
            sub.elapseTime()
            sub.observe(*observation)
            if solution_exist:
                sol.elapseTime()
                sol.observe(*observation)
            for point in get_point_gen(numRows, numCols, N2):
                pred = sub.belief.getProb(*point)
                if solution_exist:
                    answer = sol.belief.getProb(*point)
                    yield grader.requireIsEqual(answer, pred, 0.00001)
    all(get_cmp_gen())

grader.addHiddenPart('3a-2-hidden', test_3a_2, 1, description="advanced test for emission AND transition probabilities, strict time limit", maxSeconds=5)


############################################################
# Problem 4: Particle filtering

def test_4a_0():
    random.seed(3)

    pf = submission.ParticleFilter(30, 13)

    pf.observe(555, 193, 800)

    grader.requireIsEqual(0.005, pf.belief.getProb(20, 4))
    grader.requireIsEqual(0.045, pf.belief.getProb(21, 5))
    grader.requireIsEqual(0.95, pf.belief.getProb(22, 6))
    grader.requireIsEqual(0.0, pf.belief.getProb(8, 4))

    pf.observe(525, 193, 830)

    grader.requireIsEqual(0.0, pf.belief.getProb(20, 4))
    grader.requireIsEqual(0.0, pf.belief.getProb(21, 5))
    grader.requireIsEqual(1.0, pf.belief.getProb(22, 6))
    grader.requireIsEqual(0.0, pf.belief.getProb(8, 4))


grader.addBasicPart('4a-0-basic', test_4a_0, 2, description="4a basic test for PF observe")

def test_4a_1():
    random.seed(3)
    pf = submission.ParticleFilter(30, 13)
    grader.requireIsEqual(69, len(pf.particles)) # This should not fail unless your code changed the random initialization code.

    pf.elapseTime()
    grader.requireIsEqual(200, sum(pf.particles.values())) # Do not lose particles
    grader.requireIsEqual(60, len(pf.particles)) # Most particles lie on the same (row, col) locations

    grader.requireIsEqual(7, pf.particles[(3,9)])
    grader.requireIsEqual(0, pf.particles[(2,10)])
    grader.requireIsEqual(6, pf.particles[(8,4)])
    grader.requireIsEqual(0, pf.particles[(12,6)])
    grader.requireIsEqual(2, pf.particles[(7,8)])
    grader.requireIsEqual(0, pf.particles[(11,6)])
    grader.requireIsEqual(2, pf.particles[(18,7)])
    grader.requireIsEqual(1, pf.particles[(20,5)])

    pf.elapseTime()
    grader.requireIsEqual(200, sum(pf.particles.values())) # Do not lose particles
    grader.requireIsEqual(57, len(pf.particles)) # Slightly more particles lie on the same (row, col) locations

    grader.requireIsEqual(6, pf.particles[(3,9)])
    grader.requireIsEqual(0, pf.particles[(2,10)]) # 0 --> 0
    grader.requireIsEqual(7, pf.particles[(8,4)])
    grader.requireIsEqual(0, pf.particles[(12,6)])
    grader.requireIsEqual(1, pf.particles[(7,8)])
    grader.requireIsEqual(2, pf.particles[(11,6)])
    grader.requireIsEqual(0, pf.particles[(18,7)]) # 0 --> 1
    grader.requireIsEqual(3, pf.particles[(20,5)]) # 1 --> 0

grader.addBasicPart('4a-1-basic', test_4a_1, 2, description="4a basic test for PF elapseTime")

def test_4a_2():
    random.seed(3)
    pf = submission.ParticleFilter(30, 13)
    grader.requireIsEqual(69, len(pf.particles)) # This should not fail unless your code changed the random initialization code.

    pf.elapseTime()
    grader.requireIsEqual(60, len(pf.particles)) # Most particles lie on the same (row, col) locations
    pf.observe(555, 193, 800)

    grader.requireIsEqual(200, sum(pf.particles.values())) # Do not lose particles
    grader.requireIsEqual(3, len(pf.particles)) # Most particles lie on the same (row, col) locations
    grader.requireIsEqual(0.075, pf.belief.getProb(20, 4))
    grader.requireIsEqual(0.1, pf.belief.getProb(21, 5))
    grader.requireIsEqual(0.0, pf.belief.getProb(21, 6))
    grader.requireIsEqual(0.825, pf.belief.getProb(22, 6))
    grader.requireIsEqual(0.0, pf.belief.getProb(22, 7))

    pf.elapseTime()
    grader.requireIsEqual(6, len(pf.particles)) # Most particles lie on the same (row, col) locations

    pf.observe(660, 193, 50)
    grader.requireIsEqual(0.0, pf.belief.getProb(20, 4))
    grader.requireIsEqual(0.0, pf.belief.getProb(21, 5))
    grader.requireIsEqual(0.18, pf.belief.getProb(21, 6))
    grader.requireIsEqual(0.0, pf.belief.getProb(22, 6))
    grader.requireIsEqual(0.8, pf.belief.getProb(22, 7))

grader.addBasicPart('4a-2-basic', test_4a_2, 2, description="4a basic test for PF observe AND elapseTime")

grader.grade()
