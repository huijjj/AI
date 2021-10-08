#!/usr/bin/env python

import graderUtil

grader = graderUtil.Grader()
submission = grader.load('submission')

from game import Agent
from ghostAgents import RandomGhost, DirectionalGhost
import random, math, traceback, sys, os

import pacman, time, layout, textDisplay
textDisplay.SLEEP_TIME = 0
textDisplay.DRAW_EVERY = 1000
thismodule = sys.modules[__name__]

try:
    import solution
    grader.addHiddenPart = grader.addBasicPart
    SEED = solution.SEED
    solution_exist = True
except ModuleNotFoundError:
    SEED = 42
    solution_exist = False


def run(layname, pac, ghosts, nGames = 1, name = 'games', verbose=True):
  """
  Runs a few games and outputs their statistics.
  """
  if grader.fatalError:
    return {'time': 65536, 'wins': 0, 'games': None, 'scores': [0]*nGames, 'timeouts': nGames}

  starttime = time.time()
  lay = layout.getLayout(layname, 3)
  disp = textDisplay.NullGraphics()

  if verbose:
    print('*** Running %s on' % name, layname,'%d time(s).' % nGames)
  games = pacman.runGames(lay, pac, ghosts, disp, nGames, False, catchExceptions=False)
  if verbose:
    print('*** Finished running %s on' % name, layname,'after %d seconds.' % (time.time() - starttime))
  
  stats = {'time': time.time() - starttime, 'wins': [g.state.isWin() for g in games].count(True), 'games': games, 'scores': [g.state.getScore() for g in games], 'timeouts': [g.agentTimeout for g in games].count(True)}
  if verbose:
    print('*** Won %d out of %d games. Average score: %f ***' % (stats['wins'], len(games), sum(stats['scores']) * 1.0 / len(games)))

  return stats

class RecordingReflexAgent(Agent):
  def __init__(self):
    self.lastPositions = []
    self.dc = None

  def getAction(self, gameState):
    # Save the state
    recordedStates.append(gameState)

    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best

    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    return successorGameState.getScore()

recordedStates = []
hiddenTestOpponents = 2
random.seed(SEED)
run('smallClassic', RecordingReflexAgent(), [DirectionalGhost(i + 1) for i in range(hiddenTestOpponents)],
    name='recording', verbose=False)  # two ghosts


def testBasic(agentName):
  stats = {}
  if agentName == 'alphabeta':
    stats = run('smallClassic', submission.AlphaBetaAgent(depth=2), [DirectionalGhost(i + 1) for i in range(2)], name='%s (depth %d)' % ('alphabeta', 2))
  elif agentName == 'minimax':
    stats = run('smallClassic', submission.MinimaxAgent(depth=2), [DirectionalGhost(i + 1) for i in range(2)], name='%s (depth %d)' % ('minimax', 2))
  else:
    stats = run('smallClassic', submission.ExpectimaxAgent(depth=2), [DirectionalGhost(i + 1) for i in range(2)], name='%s (depth %d)' % ('expectimax', 2))
  if stats['timeouts'] > 0:
    grader.fail('Your ' + agentName + ' agent timed out on smallClassic.  No autograder feedback will be provided.')
    return
  grader.assignFullCredit()


gamePlay = {}
hiddenTestDepth = 2

def testHidden(agentFullName):
  player = 0
  depth = hiddenTestDepth
  subAgent = getattr(submission, agentFullName)(depth=depth)
  if solution_exist:
    solAgent = getattr(solution, agentFullName)(depth=depth)
    Value = getattr(solution, agentFullName[:-5] + 'Value')
    def getQ(evalFn, state, action):
      assert evalFn is not None
      succ = state.generateSuccessor(0, action)

      return Value(succ, player + 1, depth,
                   evaluationFunction=evalFn)

    num_states = 40

    for state in recordedStates[-num_states:]:
      pred = getQ(subAgent.evaluationFunction, state, subAgent.getAction(state))
      if solution_exist:
        answer = getQ(solAgent.evaluationFunction, state, solAgent.getAction(state))
        grader.requireIsEqual(answer, pred)  # compare values of successor states

    if agentFullName == 'AlphaBetaAgent':
      solMinimaxAgent = solution.MinimaxAgent(depth=depth)

      def getQValues(agent):
        return [getQ(agent.evaluationFunction, state, agent.getAction(state))
                for state in recordedStates[-num_states:]]
      tm = graderUtil.TimeMeasure()
      tm.check()
      sol_qvalues = getQValues(solMinimaxAgent)
      sol_time = tm.elapsed()

      tm.check()
      sub_qvalues = getQValues(subAgent)
      sub_time = tm.elapsed()

      print('MinimaxAgent: {} seconds'.format(sol_time))
      print('AlphaBetaAgent: {} seconds'.format(sub_time))

      grader.requireIsEqual(sol_qvalues, sub_qvalues)  # values of AlphaBetaAgent and MinimaxAgent should be same
      grader.requireIsLessThan(sol_time * 0.75, sub_time)  # AlphaBetaAgent should be faster than MinimaxAgent

maxSeconds = 10
    
grader.addBasicPart('1a-1-basic', lambda : testBasic('minimax'), 1, maxSeconds=maxSeconds, description='Tests minimax for timeout on smallClassic.')
grader.addHiddenPart('1a-2-hidden', lambda : testHidden('MinimaxAgent'), 5, maxSeconds=maxSeconds, description='Tests minimax')

grader.addBasicPart('2b-1-basic', lambda : testBasic('alphabeta'), 1, description='Tests alphabeta for timeout on smallClassic.')
grader.addHiddenPart('2b-2-hidden', lambda : testHidden('AlphaBetaAgent'), 5, maxSeconds=maxSeconds, description='Tests alphabeta')

grader.addBasicPart('3a-1-basic', lambda : testBasic('expectimax'), 1, maxSeconds=maxSeconds, description='Tests expectimax for timeout on smallClassic.')
grader.addHiddenPart('3a-2-hidden', lambda : testHidden('ExpectimaxAgent'), 5, maxSeconds=maxSeconds, description='Tests expectimax')


############################################################
# Problem 4: evaluation function

def runq4():
  """
  Runs their expectimax agent a few times and checks for victory!
  """
  random.seed(SEED)
  nGames = 20
  
  print('Running your agent %d times to compute the average score...' % nGames)
  params = '-l smallClassic -p ExpectimaxAgent -a evalFn=better -q -n %d -c' % nGames
  games = pacman.runGames(**pacman.readCommand(params.split(' ')))
  timeouts = [game.agentTimeout for game in games].count(True)
  wins = [game.state.isWin() for game in games].count(True)
  averageScore = sum(game.state.getScore() for game in games) / len(games)
  return timeouts, wins, averageScore

timeouts, wins, averageScore, firstTime = 1024, 0, 0, True

def testq4(thres):
  # We want to use the global values so we only need to compute them once
  global timeouts, wins, averageScore, firstTime

  recordScore = False
  if firstTime:
    firstTime = False
    recordScore = True
    if not grader.fatalError:
      timeouts, wins, averageScore = runq4()

  if timeouts > 0:
    grader.fail('Agent timed out on smallClassic with betterEvaluationFunction. No autograder feedback will be provided.')
  elif wins == 0: 
    grader.fail('Your better evaluation function never won any games.')
  else:
    if averageScore >= thres:
      grader.assignFullCredit()
    if recordScore:
      grader.setSide({'score': averageScore})
    
maxSeconds=300

grader.addHiddenPart('4a-1-hidden', lambda : testq4(700), 0, maxSeconds=maxSeconds, description='Check if score at least 700 on smallClassic.')
grader.addHiddenPart('4a-2-hidden', lambda : testq4(1000), 0, maxSeconds=maxSeconds, description='Check if score at least 1000 on smallClassic.')
grader.addHiddenPart('4a-3-hidden', lambda : testq4(1200), 0, maxSeconds=maxSeconds, description='Check if score at least 1200 on smallClassic.')
grader.addHiddenPart('4a-4-hidden', lambda : testq4(1400), 0, maxSeconds=maxSeconds, description='Check if score at least 1400 on smallClassic.')
grader.addHiddenPart('4a-5-hidden', lambda : testq4(1500), 5, maxSeconds=maxSeconds, description='Check if score at least 1500 on smallClassic.')

grader.grade()
