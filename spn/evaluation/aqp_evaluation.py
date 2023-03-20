import csv
import logging
import pickle
from enum import Enum
from time import perf_counter
import numpy as np
import pandas as pd
import math
import copy
import os

from ensemble_compilation.graph_representation import Query, QueryType, AggregationType, AggregationOperationType
from ensemble_compilation.physical_db import DBConnection
from ensemble_compilation.spn_ensemble import read_ensemble
from evaluation.utils import parse_query, save_csv
from spn.structure.Base import bfs
from spn.structure.Base import Product
from rspn.structure.base import Sum

logger = logging.getLogger(__name__)


class ApproachType(Enum):
    MODEL_BASED = 0
    TABLESAMPLE = 1
    VERDICT_DB = 2
    APPROXIMATE_DB = 3
    WAVELET = 4
    STRATIFIED_SAMPLING = 5


def compute_ground_truth(target_path, physical_db_name, vacuum=False, query_filename=None, query_list=None):
    """
    Queries database for each query and stores result rows in dictionary.
    :param query_filename: where to take queries from
    :param target_path: where to store dictionary
    :param physical_db_name: name of the database
    :return:
    """

    db_connection = DBConnection(db=physical_db_name)
    # read all queries
    if query_list is not None:
        queries = query_list
    elif query_filename is not None:
        with open(query_filename) as f:
            queries = f.readlines()
    else:
        raise ValueError("Either query_list or query_filename have to be given")

    ground_truth = dict()
    ground_truth_times = dict()

    for query_no, query_str in enumerate(queries):

        query_str = query_str.strip()
        logger.debug(f"Computing ground truth for AQP query {query_no}: {query_str}")

        aqp_start_t = perf_counter()
        query_str = query_str.strip()
        rows = db_connection.get_result_set(query_str)
        ground_truth[query_no] = rows
        aqp_end_t = perf_counter()
        ground_truth_times[query_no] = aqp_end_t - aqp_start_t
        logger.info(f"\t\ttotal time for query execution: {aqp_end_t - aqp_start_t} secs")

        if vacuum:
            vacuum_start_t = perf_counter()
            db_connection.vacuum()
            vacuum_end_t = perf_counter()
            logger.info(f"\t\tvacuum time: {vacuum_end_t - vacuum_start_t} secs")

        dump_ground_truth(ground_truth, ground_truth_times, target_path)
    dump_ground_truth(ground_truth, ground_truth_times, target_path)


def dump_ground_truth(ground_truth, ground_truth_times, target_path):
    with open(target_path, 'wb') as f:
        logger.debug(f"\t\tSaving ground truth dictionary to {target_path}")
        pickle.dump(ground_truth, f, pickle.HIGHEST_PROTOCOL)
    with open(target_path + '_times.pkl', 'wb') as f:
        logger.debug(f"\t\tSaving ground truth dictionary times to {target_path}")
        pickle.dump(ground_truth_times, f, pickle.HIGHEST_PROTOCOL)


def compute_relative_error(true, predicted, debug=False):
    true = float(true)
    predicted = float(predicted)
    relative_error = (true - predicted) / true
    if debug:
        logger.debug(f"\t\tpredicted     : {predicted:.2f}")
        logger.debug(f"\t\ttrue          : {true:.2f}")
        logger.debug(f"\t\trelative_error: {100 * relative_error:.2f}%")
    return abs(relative_error)


def evaluate_aqp_queries(return_node_status, target_ns_path, ensemble_location, query_filename, target_path, schema, ground_truth_path,
                         rdc_spn_selection, pairwise_rdc_path, max_variants=5, merge_indicator_exp=False,
                         exploit_overlapping=False, min_sample_ratio=0, debug=False,
                         show_confidence_intervals=False):
    """
    Loads ensemble and computes metrics for AQP query evaluation
    :param ensemble_location:
    :param query_filename:
    :param target_csv_path:
    :param schema:
    :param max_variants:
    :param merge_indicator_exp:
    :param exploit_overlapping:
    :param min_sample_ratio:
    :return:
    """

    spn_ensemble = read_ensemble(ensemble_location, build_reverse_dict=True)
    
    csv_rows = []
    node_status = []

    # read all queries
    with open(query_filename) as f:
        queries = f.readlines()
    # read ground truth
    with open(ground_truth_path, 'rb') as handle:
        ground_truth = pickle.load(handle)

    for query_no, query_str in enumerate(queries):
        
        query_str = query_str.strip()
        logger.info(f"Evaluating AQP query {query_no}: {query_str}")

        # parse query
        query = parse_query(query_str.strip(), schema)
        aqp_start_t = perf_counter()
        
        confidence_intervals, aqp_result, ns = spn_ensemble.evaluate_query(return_node_status, query_no, query, rdc_spn_selection=rdc_spn_selection,
                                                                    pairwise_rdc_path=pairwise_rdc_path,
                                                                    merge_indicator_exp=merge_indicator_exp,
                                                                    max_variants=max_variants,
                                                                    exploit_overlapping=exploit_overlapping,
                                                                    debug=debug,
                                                                    confidence_intervals=show_confidence_intervals)

        aqp_end_t = perf_counter()
        latency = aqp_end_t - aqp_start_t
        logger.info(f"\t\t{'total_time:':<32}{latency} secs")

        if ground_truth is not None:
            print("aqp_evaluation ground_truth length:", len(ground_truth))
            true_result = ground_truth[query_no]
            
            if isinstance(aqp_result, list):
                average_relative_error, bin_completeness, false_bin_percentage, total_bins, \
                confidence_interval_precision, confidence_interval_length, _ = \
                    evaluate_group_by(aqp_result, true_result, confidence_intervals)

                logger.info(f"\t\t{'total_bins: ':<32}{total_bins}")
                logger.info(f"\t\t{'bin_completeness: ':<32}{bin_completeness * 100:.2f}%")
                logger.info(f"\t\t{'average_relative_error: ':<32}{average_relative_error * 100:.2f}%")
                logger.info(f"\t\t{'false_bin_percentage: ':<32}{false_bin_percentage * 100:.2f}%")
                if show_confidence_intervals:
                    logger.info(
                        f"\t\t{'confidence_interval_precision: ':<32}{confidence_interval_precision * 100:>.2f}%")
                    logger.info(f"\t\t{'confidence_interval_length: ':<32}{confidence_interval_length * 100:>.2f}%")

            else:
                true_result = true_result[0][0]
                predicted_value = aqp_result

                logger.info(f"\t\t{'predicted:':<32}{predicted_value}")
                logger.info(f"\t\t{'true:':<32}{true_result}")
                # logger.info(f"\t\t{'confidence_interval:':<32}{confidence_intervals}")
                relative_error = compute_relative_error(true_result, predicted_value)
                logger.info(f"\t\t{'relative_error:':<32}{relative_error * 100:.2f}%")
                if show_confidence_intervals:
                    confidence_interval_precision, confidence_interval_length = evaluate_confidence_interval(
                        confidence_intervals,
                        true_result,
                        predicted_value)
                    logger.info(
                        f"\t\t{'confidence_interval_precision:':<32}{confidence_interval_precision * 100:>.2f}")
                    logger.info(f"\t\t{'confidence_interval_length: ':<32}{confidence_interval_length * 100:>.2f}%")
                total_bins = 1
                bin_completeness = 1
                average_relative_error = relative_error
            csv_rows.append({'approach': ApproachType.MODEL_BASED,
                            'query_no': query_no,
                            'latency': latency,
                            'average_relative_error': average_relative_error * 100,
                            'bin_completeness': bin_completeness * 100,
                            'total_bins': total_bins,
                            'query': query_str,
                            'sample_percentage': 100
                            })
        else:
            logger.info(f"\t\tpredicted: {aqp_result}")

        node_status.append(ns)

    save_csv(csv_rows, target_path)
    with open(target_ns_path, 'wb') as f:
        pickle.dump(node_status, target_ns_path, pickle.HIGHEST_PROTOCOL)


def evaluate_confidence_interval(confidence_interval, true_result, predicted):
    in_interval = 0
    if confidence_interval[0] <= true_result <= confidence_interval[1]:
        in_interval = 1
    relative_interval_size = (confidence_interval[1] - predicted) / predicted
    return in_interval, relative_interval_size


def evaluate_group_by(aqp_result, true_result, confidence_intervals, medians=False, debug=False):
    group_by_combinations_found = 0
    avg_relative_errors = []
    confidence_interval_precision = 0
    confidence_interval_length = 0

    print("aqp_evaluation true_result:", true_result)
    for result_row in true_result:
        # print("aqp_evaluation result_row:", result_row)
        group_by_attributes = result_row[:-1]
        # print("aqp_evaluation aqp_result:", aqp_result)
        # print("group_by_attributes:", group_by_attributes)
        # print("type of group_by_attributes:", type(group_by_attributes))
        # for matching_idx, aqp_row in enumerate(aqp_result):
        #     print("type of aqp_row:", type(aqp_row))
        #     print("aqp_row:", aqp_row)
        #     print("aqp_row[:-1] to tuple:", tuple(aqp_row)[:-1])
        #     print("type of aqp_row[:-1]:", type(aqp_row[:-1]))


        # 经过之前的变换aqp_row变成了np.record类型，不能用[:-1]的方式切片，需要转为tuple类型
        matching_aqp_rows = [(matching_idx, aqp_row) for matching_idx, aqp_row in enumerate(aqp_result)
                             if tuple(aqp_row)[:-1] == group_by_attributes]
        assert len(matching_aqp_rows) <= 1, "Multiple possible group by attributes found."
        if len(matching_aqp_rows) == 1:
            matching_idx = matching_aqp_rows[0][0]
            matching_aqp_row = matching_aqp_rows[0][1]

            group_by_combinations_found += 1
            assert tuple(matching_aqp_row)[:-1] == result_row[:-1]
            relative_error = compute_relative_error(result_row[-1], matching_aqp_row[-1], debug=debug)
            avg_relative_errors.append(relative_error)
            if confidence_intervals:
                in_interval, relative_interval_size = evaluate_confidence_interval(confidence_intervals[matching_idx],
                                                                                   result_row[-1],
                                                                                   matching_aqp_row[-1])
                confidence_interval_precision += in_interval
                confidence_interval_length += relative_interval_size

    bin_completeness = math.inf
    average_relative_error = math.inf
    false_bin_percentage = math.inf
    total_bins = len(true_result)
    if group_by_combinations_found > 0:
        bin_completeness = group_by_combinations_found / len(true_result)
        if not medians:
            average_relative_error = sum(avg_relative_errors) / group_by_combinations_found
        else:
            average_relative_error = np.median(avg_relative_errors)
        false_bin_percentage = (len(aqp_result) - group_by_combinations_found) / len(aqp_result)
        confidence_interval_precision /= group_by_combinations_found
        confidence_interval_length /= group_by_combinations_found

    max_error = math.inf if len(avg_relative_errors) == 0 else max(avg_relative_errors)
    return average_relative_error, bin_completeness, false_bin_percentage, total_bins, confidence_interval_precision, confidence_interval_length, max_error

"""
def stratified_evaluate(true_result, aqp_result, schema, query_str, query):

    '''
    # store the sampleratio data into .pkl files (dataframe format)
    # ss1
    sql = 'select groupname, ratio from sampleratio where samplename=\'flights_stratified\';'
    db_connection = DBConnection(db='flights')
    aqp_start_t = perf_counter()
    rows = db_connection.get_result_set(sql)
    aqp_end_t = perf_counter()
    print("rows:", rows)
    df = pd.DataFrame(rows, columns=['groupname','ratio'])
    print("dataframe:", df)
    #df = pd.concat([df['groupname'].str.split(',',expand=True), df['ratio']], axis=1) 
    #df.columns=['year_date', 'unique_carrier', 'ratio']
    #print("dataframe:", df)
    with open("./benchmarks/flights/ss_ratios/ss1_ratio.pkl", 'wb+') as f:
        pickle.dump(df, f, pickle.HIGHEST_PROTOCOL)

    # ss2
    sql = 'select groupname, ratio from sampleratio where samplename=\'flights_stratified_2\';'
    db_connection = DBConnection(db='flights')
    aqp_start_t = perf_counter()
    rows = db_connection.get_result_set(sql)
    aqp_end_t = perf_counter()
    print("rows:", rows)
    df = pd.DataFrame(rows, columns=['groupname','ratio'])
    print("dataframe:", df)
    with open("./benchmarks/flights/ss_ratios/ss2_ratio.pkl", 'wb+') as f:
        pickle.dump(df, f, pickle.HIGHEST_PROTOCOL)

    # ss3
    sql = 'select groupname, ratio from sampleratio where samplename=\'flights_stratified_3\';'
    db_connection = DBConnection(db='flights')
    aqp_start_t = perf_counter()
    rows = db_connection.get_result_set(sql)
    aqp_end_t = perf_counter()
    print("rows:", rows)
    df = pd.DataFrame(rows, columns=['groupname','ratio'])
    print("dataframe:", df)
    #df = pd.concat([df['groupname'].str.split(',',expand=True), df['ratio']], axis=1) 
    #df.columns=['dest', 'origin_state_abr', 'ratio']
    #print("dataframe:", df)
    with open("./benchmarks/flights/ss_ratios/ss3_ratio.pkl", 'wb+') as f:
        pickle.dump(df, f, pickle.HIGHEST_PROTOCOL)
    '''

    from schemas.flights.schema import gen_flights_500M_stratified_1_schema, gen_flights_500M_stratified_2_schema, gen_flights_500M_stratified_3_schema
    group_attrs = []
    ssname = 'ss1'
    samplename = ''
    sampleratios = None
    # 1. to decide which schema
    if ssname=='ss1':
        samplename = 'flights_stratified'
        group_attrs.append('year_date')
        group_attrs.append('unique_carrier')
        with open("./benchmarks/flights/ss_ratios/ss1_ratio.pkl", 'rb') as f:
            sampleratios = pickle.load(f)

    elif ssname=='ss2':
        samplename = 'flights_stratified_2'
        group_attrs.append('origin')
        with open("./benchmarks/flights/ss_ratios/ss2_ratio.pkl", 'rb') as f:
            sampleratios = pickle.load(f)
    elif ssname=='ss3':
        samplename = 'flights_stratified_3'
        group_attrs.append('origin_state_abr')
        group_attrs.append('dest')
        with open("./benchmarks/flights/ss_ratios/ss3_ratio.pkl", 'rb') as f:
            sampleratios = pickle.load(f)
    else:
        print("No such schema")

    print("stratified evaluation - true result:", true_result)
    print("stratified evaluation - aqp result:", aqp_result)
    common_group = list(set(group_attrs).intersection(set(list(dict(query.group_bys).values()))))
    print("stratified evaluation - group_attrs:", group_attrs)
    print("stratified evaluation - query.group_bys:", list(dict(query.group_bys).values()))
    print("stratified evaluation - common_group:", common_group)
    if len(common_group)==0:
        return aqp_result
    else:
        # 如果聚合类型是COUNT类型或者SUM，并且group by其中任一或多个属性，需要除以对应的sample ratio
        if query.query_type == QueryType.CARDINALITY or any(
                [aggregation_type == AggregationType.SUM or aggregation_type == AggregationType.COUNT
                for _, aggregation_type, _ in query.aggregation_operations]):
            unbiased_aqp_result = []
            # 查询sampleratio
            #sampleratio = sampleratios.groupby(common_group[0]).agg({'ratio': 'sum'})
            print("sampleratios:", sampleratios)
            #sampleratio.to_dict('list')
            for item in aqp_result:
                groupname = item[0]
                if 'year_date' in common_group:
                    groupname = str(int(groupname))
                value = item[1]
                print("groupname:", groupname)
                print("value:", value)
                a = sampleratios.loc[sampleratios['groupname']==groupname]['ratio'].values[0]
                print("a:", a)
                #print("selected:", sampleratios[sampleratios.loc[common_group[0]==groupname]]['ratio'])
                unbiased_value = item[1]/(100*float(a))
                unbiased_aqp_result.append((groupname, unbiased_value))
            return unbiased_aqp_result
        else:
            return aqp_result
    return aqp_result
"""