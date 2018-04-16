class Average:

    def __init__(self):
        self.mean, self.m2, self.var, self.count = 0.0, 0.0, 1e10, 0

    def reset(self):
        self.mean, self.m2, self.var, self.count = 0.0, 0.0, 1e10, 0

    def update(self, point):

        # taken from The Art of Computer Programming (Vol II) by D. Knuth
        # online variance formula with complexity O(1) per sample
        # also safeguarded against numerical instability
        self.count += 1
        count = self.count
        delta = point - self.mean
        self.mean += delta / count
        self.m2 += delta * (point - self.mean)
        if count > 1:
            self.var = self.m2 / (count - 1.0)
        return self.var
