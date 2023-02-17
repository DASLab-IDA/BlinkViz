import logging
from spn.ensemble_compilation.spn_ensemble import SPNEnsemble
from spn.ensemble_compilation.graph_representation import SchemaGraph
from res_nn.core.Models.BaseModel import mlps
from spn.evaluation.utils import parse_query, save_csv
from time import perf_counter

logger = logging.getLogger(__name__)

class Ensemble():
    def __init__(self, spn_ensembles, res_nn, schema_graph):
        self.spn_ensembles = spn_ensembles # SPNEnsemble list
        self.res_nn = res_nn
        self.schema_graph = schema_graph

    def evaluate_spn_ensemble(self):
        pass

    def evaluate_res_nn(self):
        pass

    def evaluate_together_single_query(self, query):
        
        # evaluate by each spn_ensemble, get the preliminary answers and status vectors
        preds, status_vectors, spn_latency = [], [], []

        # 之后改成开多个线程，不然跟结果对应不起来
        for spn_ensemble in self.spn_ensembles:
            query = parse_query(query.strip(), self.schema_graph)
            aqp_start_t = perf_counter()
        
            # 之后让status vector和相应group的projection都在这里返回，而不再存储到文件中了
            confidence_intervals, aqp_result = spn_ensemble.evaluate_query(query_no, _path, query, rdc_spn_selection=rdc_spn_selection,
                                                                    pairwise_rdc_path=pairwise_rdc_path,
                                                                    merge_indicator_exp=merge_indicator_exp,
                                                                    max_variants=max_variants,
                                                                    exploit_overlapping=exploit_overlapping,
                                                                    debug=debug,
                                                                    confidence_intervals=show_confidence_intervals)

            aqp_end_t = perf_counter()
            latency = aqp_end_t - aqp_start_t
            logger.info(f"\t\t{'spn_time:':<32}{latency} secs")

        # 处理projection

        # concecate together as the input of res_nn, output the refined results
        # 有问题
        inputs = []
        input = []
        for sv in status_vectors:
            for each_sv in sv:
                input.append(each_sv)
        for pred in preds:
            input.append(pred)
        inputs.append(input)

        for input in inputs:
            evaluate_nn()