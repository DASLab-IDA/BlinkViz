from spn.ensemble_compilation.spn_ensemble import SPNEnsemble
from spn.ensemble_compilation.graph_representation import SchemaGraph
from res_nn.core.Models.BaseModel import mlps

class Ensemble():
    def __init__(self, spn_ensemble, mlps, schema_graph):
        self.spn_ensemble = spn_ensemble
        self.mlps = mlps