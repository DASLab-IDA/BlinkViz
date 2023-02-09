from spn.ensemble_compilation.spn_ensemble import SPNEnsemble
from spn.ensemble_compilation.graph_representation import SchemaGraph
from res_nn.core.Models.BaseModel import mlps

class Ensemble():
    def __init__(self, spn_ensembles, res_nn, schema_graph):
        self.spn_ensembles = spn_ensembles # SPNEnsemble list
        self.res_nn = res_nn
        self.schema_graph = schema_graph

    def evaluate_spn_ensemble(self):
        pass

    def evaluate_res_nn(self):
        pass

    def evaluate_together(self):
        pass
