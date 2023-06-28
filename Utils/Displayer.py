class Displayer:
    def __init__(self, name_list, full_count):
        self.full_count = full_count
        self.name_list = name_list
        self.loss_score_list = [0] * len(self.name_list)

    def record(self, values):
        for i, value in enumerate(values):
            self.loss_score_list[i] += value.item()

    def get_avg(self, full_count):
        return [value/full_count for value in self.loss_score_list]

    def reset(self):
        self.loss_score_list = [0] * len(self.name_list)