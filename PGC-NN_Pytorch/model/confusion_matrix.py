
class ConfusionMatrix:

    def __init__(self, poi_type='all types'):
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0
        self.poi_type = poi_type
        self.total_samples_of_poi_type = 0
        self.total_users_inverted_routine_tp = 0

    def add_tp(self):
        self.t p+ =1

    def add_fp(self):
        self.f p+ =1

    def add_tn(self):
        self.t n+ =1

    def add_fn(self):
        self.f n+ =1

    def classification_report(self):
        precision = self.t p /(self.tp + self.fp)
        recall = self.t p /(self.tp + self.fn)
        fscore = 2* (precision * recall) / (precision + recall)
        print("---------")
        print("\nPoi type: ", self.poi_type)
        print("\nPrecision: ", precision)
        print("\nRecall: ", recall)
        print("\nF1-score: ", fscore)
        if self.total_users_inverted_routine_tp > 0:
            print("\nHits from users that have inverted routine")
            print("\nQuantidade: ", self.total_users_inverted_routine_tp)

    def set_total_samples_of_poi_type(self, total):
        self.total_samples_of_poi_type += total

    def add_total_users_inverted_routine_tp(self):
        self.total_users_inverted_routine_tp += 1

    def results(self):
        precision = self.tp / (self.tp + self.fp)
        recall = self.tp / (self.tp + self.fn)
        if (precision + recall) != 0:
            fscore = 2 * (precision * recall) / (precision + recall)
        else:
            fscore = 0
        return precision, recall, fscore