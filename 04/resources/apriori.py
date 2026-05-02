from collections import Counter

def frequentItems(transactions, support):
    counter = Counter()
    for trans in transactions:
        counter.update(frozenset([t]) for t in trans)
    return set(item for item in counter if counter[item]/len(transactions) >= support), counter

def generateCandidates(L, k):
    candidates = set()
    for a in L:
        for b in L:
            union = a | b
            if len(union) == k and a != b:
                candidates.add(union)
    return candidates

def filterCandidates(transactions, itemsets, support):
    counter = Counter()
    for trans in transactions:
        subsets = [itemset for itemset in itemsets if itemset.issubset(trans)]
        counter.update(subsets)
    return set(item for item in counter if counter[item]/len(transactions) >= support), counter

def apriori(transactions, support, max_len=None):
    result = list()
    resultc = Counter()
    candidates, counter = frequentItems(transactions, support)
    result += candidates
    resultc += counter
    k = 2
    while candidates:
        if max_len is not None and k > max_len:
            break
        candidates = generateCandidates(candidates, k)
        candidates,counter = filterCandidates(transactions, candidates, support)
        result += candidates
        resultc += counter
        k += 1
    resultc = {item:(resultc[item]/len(transactions)) for item in resultc}
    return result, resultc

def genereateRules(frequentItemsets, supports, minConfidence):
    result = []
    for fset in frequentItemsets:
        for consequent in fset:
            antecedent = set(fset).difference([consequent])
            support = supports[fset]
            confidence = supports[fset]
            if len(antecedent) > 0:
                confidence = support/supports[frozenset(antecedent)]
            if confidence > minConfidence:
                result.append({
                    "antecedent": antecedent,
#                     "consequent": set([consequent]),
                    "consequent": consequent,
                    "len": len(fset),
                    "support": support,
                    "confidence": confidence
                })
    return pd.DataFrame(result)