def accuracy(pred, target):
    return sum(pred == target) / len(pred)
