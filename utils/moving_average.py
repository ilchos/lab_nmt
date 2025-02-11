class MovingAverage:
    def __init__(self, window_size):
        self.window_size = window_size
        self.values = []
        self.total = 0.0

    def update(self, new_value):
        self.values.append(new_value)
        self.total += new_value
        if len(self.values) > self.window_size:
            self.total -= self.values.pop(0)

    def get_moving_average(self):
        if not self.values:
            return 0.0
        return self.total / len(self.values)

    def __call__(self, new_value):
        self.update(new_value)
        return self.get_moving_average()

    @property
    def value(self):
        return self.get_moving_average()
