from exploration.Epsilon import Epsilon


# a generic epsilon strategy based on an externally provided function
class GenericEpsilon(Epsilon):

    def __init__(self, generator):
        super().__init__(generator())
        self.generator = generator

    def next_epsilon(self, current_epsilon, epoch):
        return self.generator()
