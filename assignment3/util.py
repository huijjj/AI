import os, random, operator, sys
from collections import Counter


def dotProduct(d1, d2):
    """
    @param dict d1: a feature vector represented by a mapping from a feature (string) to a weight (float).
    @param dict d2: same as d1
    @return float: the dot product between d1 and d2
    """
    if len(d1) < len(d2):
        return dotProduct(d2, d1)
    else:
        return sum(d1.get(f, 0) * v for f, v in list(d2.items()))


def increment(d1, scale, d2):
    """
    Implements d1 += scale * d2 for sparse vectors.
    @param dict d1: the feature vector which is mutated.
    @param float scale
    @param dict d2: a feature vector.
    """
    for f, v in list(d2.items()):
        d1[f] = d1.get(f, 0) + v * scale


def readExamples(path):
    """
    Reads a set of training examples.
    """
    examples = []
    for line in open(path, encoding="ISO-8859-1"):
        # Format of each line: <output label (+1 or -1)> <input sentence>
        y, x = line.split(" ", 1)
        examples.append((x.strip(), int(y)))
    print("Read %d examples from %s" % (len(examples), path))
    return examples


def evaluatePredictor(examples, predictor):
    """
    predictor: a function that takes an x and returns a predicted y.
    Given a list of examples (x, y), makes predictions based on |predict| and returns the fraction
    of misclassiied examples.
    """
    error = 0
    for x, y in examples:
        if predictor(x) != y:
            error += 1
    return 1.0 * error / len(examples)


def outputWeights(weights, path):
    print("%d weights" % len(weights))
    out = open(path, "w", encoding="utf-8")
    for f, v in sorted(list(weights.items()), key=lambda f_v: -f_v[1]):
        print("\t".join([str(f), str(v)]), file=out)
    out.close()


def verbosePredict(phi, y, weights, out):
    yy = 1 if dotProduct(phi, weights) > 0 else -1
    if y:
        print(
            "Truth: %s, Prediction: %s [%s]" % (y, yy, "CORRECT" if y == yy else "WRONG"),
            file=out,
        )
    else:
        print("Prediction:", yy, file=out)
    for f, v in sorted(
        list(phi.items()), key=lambda f_v1: -f_v1[1] * weights.get(f_v1[0], 0)
    ):
        w = weights.get(f, 0)
        print("%-30s%s * %s = %s" % (f, v, w, v * w), file=out)
    return yy


def outputErrorAnalysis(examples, featureExtractor, weights, path):
    out = open(path, "w", encoding="utf-8")
    for x, y in examples:
        print("===", x, file=out)
        verbosePredict(featureExtractor(x), y, weights, out)
    out.close()


def interactivePrompt(featureExtractor, weights):
    while True:
        print("> ", end=" ")
        x = sys.stdin.readline()
        if not x:
            break
        phi = featureExtractor(x)
        verbosePredict(phi, None, weights, sys.stdout)
