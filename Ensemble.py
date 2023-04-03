import sys
#sys.path.append("..")
#sys.path.append("/home/qym/github_keeping/BlinkViz/spn")
#sys.path.append("./spn/ensemble_compilation")
#sys.path.append("/home/qym/github_keeping/BlinkViz/spn/ensemble_compilation")
import logging
import os
import time
from mspn.ensemble_compilation.spn_ensemble import SPNEnsemble
from mspn.ensemble_compilation.graph_representation import SchemaGraph
from res_nn.core.Models.BaseModel import mlps
from mspn.evaluation.utils import parse_query, save_csv
from mspn.schemas.flights.schema import gen_flights_500M_schema
from mspn.schemas.ssb.schema import gen_sf50_ssb_schema
from time import perf_counter
from mspn.ensemble_compilation.spn_ensemble import read_ensemble


logger = logging.getLogger(__name__)

class Ensemble():
    def __init__(self, spn_ensembles, res_nn, dataset):
        self.spn_ensembles = spn_ensembles # SPNEnsemble location list
        self.res_nn = res_nn
        self.dataset = dataset
        
    
        # Generate schema
        table_csv_path = ''
        if dataset == 'ssb-sf50':
            self.schema = gen_sf50_ssb_schema(table_csv_path)
        elif dataset == 'flights500M':
            self.schema = gen_flights_500M_schema(table_csv_path)
        else:
            raise ValueError('Dataset unknown')


    def evaluate_spn_ensemble(self):
        pass

    def evaluate_res_nn(self):
        pass

    def evaluate_together_single_query(self, query):
        os.makedirs('logs', exist_ok=True)
        logging.basicConfig(
            level=logging.DEBUG,
            # [%(threadName)-12.12s]
            format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
            handlers=[
                logging.FileHandler("logs/{}_{}.log".format(self.dataset, time.strftime("%Y%m%d-%H%M%S"))),
                logging.StreamHandler()
            ])

        # evaluate by each spn_ensemble, get the preliminary answers and status vectors
        preds, status_vectors, spn_latency = [], [], []

        # 之后改成开多个线程，不然跟结果对应不起来
        for spn_ensemble in self.spn_ensembles:
            query = parse_query(query.strip(), self.schema)
            spn = read_ensemble(spn_ensemble, build_reverse_dict=True)
            aqp_start_t = perf_counter()
        
            # 之后让status vector和相应group的projection都在这里返回，而不再存储到文件中了
            confidence_intervals, aqp_result, status_vector = spn.evaluate_query(True, 1, query,
                                                                    rdc_spn_selection=False,
                                                                    pairwise_rdc_path=None,
                                                                    merge_indicator_exp=False,
                                                                    max_variants=5,
                                                                    exploit_overlapping=False,
                                                                    debug=False,
                                                                    confidence_intervals=False)

            aqp_end_t = perf_counter()
            latency = aqp_end_t - aqp_start_t
            logger.info(f"\t\t{'spn_time:':<32}{latency} secs")
            logger.info(f"\t\t{'aqp_result:'}{aqp_result}")
            logger.info(f"\t\t{'status_vector:'}{status_vector}")
            exit()

            # 处理projection

            # 记录
            preds.append(aqp_result)
            status_vectors.append(status_vectors)
            spn_latency.append(latency)

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
        
