import logging
from time import perf_counter
import time
import os

import numpy as np
import pickle
from spn.algorithms.Inference import likelihood
from spn.structure.Base import Product

from ..code_generation.convert_conditions import convert_range
from ..structure.base import Sum

logger = logging.getLogger(__name__)


def expectation(return_node_status, spn, feature_scope, inverted_features, ranges, node_expectation=None, node_likelihoods=None,
                use_generated_code=False, spn_id=None, meta_types=None, gen_code_stats=None):
    """Compute the Expectation:
        E[1_{conditions} * X_feature_scope]
        First factor is one if condition is fulfilled. For the second factor the variables in feature scope are
        multiplied. If inverted_features[i] is True, variable is taken to denominator.
        The conditional expectation would be E[1_{conditions} * X_feature_scope]/P(conditions)
    """
    
    # evidence_scope = set([i for i, r in enumerate(ranges) if not np.isnan(r)])
    evidence_scope = set([i for i, r in enumerate(ranges[0]) if r is not None])
    evidence = ranges

    assert not (len(evidence_scope) > 0 and evidence is None)

    relevant_scope = set()
    relevant_scope.update(evidence_scope)
    relevant_scope.update(feature_scope)
    if len(relevant_scope) == 0:
        
        return np.ones((ranges.shape[0], 1))

    if ranges.shape[0] == 1:

        applicable = True
        if use_generated_code:
            boolean_relevant_scope = [i in relevant_scope for i in range(len(meta_types))]
            boolean_feature_scope = [i in feature_scope for i in range(len(meta_types))]
            applicable, parameters = convert_range(boolean_relevant_scope, boolean_feature_scope, meta_types, ranges[0],
                                                   inverted_features)

        # generated C++ code
        if use_generated_code and applicable:
            time_start = perf_counter()
            import optimized_inference

            spn_func = getattr(optimized_inference, f'spn{spn_id}')
            result = np.array([[spn_func(*parameters)]])

            time_end = perf_counter()

            if gen_code_stats is not None:
                gen_code_stats.calls += 1
                gen_code_stats.total_time += (time_end - time_start)

            # logger.debug(f"\t\tGenerated Code Latency: {(time_end - time_start) * 1000:.3f}ms")
           
            return result

        # lightweight non-batch version
        else:
        
            node_status = dict()
     
            result, ns = expectation_recursive(return_node_status, spn, feature_scope, inverted_features, relevant_scope, evidence,
                                        node_expectation, node_likelihoods, node_status)
            result = np.array([result])
            print("node_status - exp_recursive:", ns)
            return result, ns
    # full batch version
    node_status = dict()
    result, ns = expectation_recursive_batch(return_node_status, spn, feature_scope, inverted_features, relevant_scope, evidence,
                                       node_expectation, node_likelihoods, node_status)
    print("node_status - exp_recursive_batch:", ns)

    return result, ns

def expectation_recursive_batch(return_node_status, node, feature_scope, inverted_features, relevant_scope, evidence, node_expectation,
                                node_likelihoods, node_status):
    if isinstance(node, Product):
        print("exp_batch - product - node:", node.id)

        llchildren = np.concatenate(
            [expectation_recursive_batch(return_node_status, child, feature_scope, inverted_features, relevant_scope, evidence,
                                         node_expectation, node_likelihoods, node_status)[0]
             for child in node.children if
             len(relevant_scope.intersection(child.scope)) > 0], axis=1)
        result = np.nanprod(llchildren, axis=1).reshape(-1, 1)

        print("exp_batch - product - node_status:", result)
        
        if node.id in list(node_status.keys()):
            node_status[node.id].append(result)
        else:
            node_status[node.id] = []
            node_status[node.id].append(result)
        return result, node_status

    elif isinstance(node, Sum):
        print("exp_batch - sum - node:", node.id)
        if len(relevant_scope.intersection(node.scope)) == 0:
            result = np.full((evidence.shape[0], 1), np.nan)
            
            if node.id in list(node_status.keys()):
                node_status[node.id].append(result)
            else:
                node_status[node.id] = []
                node_status[node.id].append(result)
            print("exp_batch - sum - node_status:", result)
            return result, node_status

        llchildren = np.concatenate(
            [expectation_recursive_batch(return_node_status, child, feature_scope, inverted_features, relevant_scope, evidence,
                                         node_expectation, node_likelihoods, node_status)[0]
             for child in node.children], axis=1)

        relevant_children_idx = np.where(np.isnan(llchildren[0]) == False)[0]
        if len(relevant_children_idx) == 0:
            if node.id in list(node_status.keys()):
                node_status[node.id].append(np.array([np.nan]))
            else:
                node_status[node.id] = []
                node_status[node.id].append(np.array([np.nan]))
        
            print("exp_batch - sum - node_status:", np.array([np.nan]))
            return np.array([np.nan]), node_status

        weights_normalizer = sum(node.weights[j] for j in relevant_children_idx)
        b = np.array(node.weights)[relevant_children_idx] / weights_normalizer

        result = np.dot(llchildren[:, relevant_children_idx], b).reshape(-1, 1)
        
        print("exp_batch - sum - node_status:", result)
        if node.id in list(node_status.keys()):
            node_status[node.id].append(result)
        else:
            node_status[node.id] = []
            node_status[node.id].append(result)
        return result, node_status

    else:
        print("exp_batch - leaf - node:", node.id)
        if node.scope[0] in feature_scope:
            t_node = type(node)
            if t_node in node_expectation:
                exps = np.zeros((evidence.shape[0], 1))

                feature_idx = feature_scope.index(node.scope[0])
                inverted = inverted_features[feature_idx]

                exps[:] = node_expectation[t_node](node, evidence, inverted=inverted)
            
                print("exp_batch - leaf - node_expectation:", exps)
                
                if node.id in list(node_status.keys()):
                    node_status[node.id].append(exps)
                else:
                    node_status[node.id] = []
                    node_status[node.id].append(exps)
                return exps, node_status
            else:
                raise Exception('Node type unknown: ' + str(t_node))

        result = likelihood(node, evidence, node_likelihood=node_likelihoods)
    
        print("exp_batch - leaf - node_likelihood:", result)
        
        if node.id in list(node_status.keys()):
            node_status[node.id].append(result)
        else:
            node_status[node.id] = []
            node_status[node.id].append(result)
        return result, node_status


def nanproduct(product, factor):
    if np.isnan(product):
        if not np.isnan(factor):
            return factor
        else:
            return np.nan
    else:
        if np.isnan(factor):
            return product
        else:
            return product * factor


def expectation_recursive(return_node_status, node, feature_scope, inverted_features, relevant_scope, evidence, node_expectation,
                          node_likelihoods, node_status):
    if isinstance(node, Product):
        print("exp - product - node:", node.id)
        product = np.nan
        for child in node.children:
            if len(relevant_scope.intersection(child.scope)) > 0:
                factor, node_status = expectation_recursive(return_node_status, child, feature_scope, inverted_features, relevant_scope, evidence,
                                               node_expectation, node_likelihoods, node_status)
                product = nanproduct(product, factor)
                print("exp - product - children:", child.id)
                print("exp - product - factor:", factor)
                print("exp - product - node_status:", node_status)
                
                
        if node.id in list(node_status.keys()):
            node_status[node.id].append(product)
        else:
            node_status[node.id] = []
            node_status[node.id].append(product)
        return product, node_status

    elif isinstance(node, Sum):
        print("exp - sum - node:", node.id)
        if len(relevant_scope.intersection(node.scope)) == 0:
            
            if node.id in list(node_status.keys()):
                node_status[node.id].append(np.nan)
            else:
                node_status[node.id] = []
                node_status[node.id].append(np.nan)
            print("exp - sum - node_status:", np.nan)
            return np.nan, node_status

        llchildren = []
        for child in node.children:
            c, ns = expectation_recursive(return_node_status, child, feature_scope, inverted_features, relevant_scope, evidence,
                                            node_expectation, node_likelihoods, node_status)
            llchildren.append(c)
            print("exp - sum - children:", child.id)
            print("exp - sum - c:", c)
            print("exp - sum - node_status:", ns)

        relevant_children_idx = np.where(np.isnan(llchildren) == False)[0]

        if len(relevant_children_idx) == 0:
            
            if node.id in list(node_status.keys()):
                node_status[node.id].append(np.nan)
            else:
                node_status[node.id] = []
                node_status[node.id].append(np.nan)
            print("exp - sum - node_status:", np.nan)
            return np.nan, node_status

        weights_normalizer = sum(node.weights[j] for j in relevant_children_idx)
        weighted_sum = sum(node.weights[j] * llchildren[j] for j in relevant_children_idx)

        result = weighted_sum / weights_normalizer

        print("exp - sum - node_status:", result)
        
        if node.id in list(node_status.keys()):
            node_status[node.id].append(result)
        else:
            node_status[node.id] = []
            node_status[node.id].append(result)
        return result, node_status

    else:
        print("exp - leaf - node:", node.id)
        if node.scope[0] in feature_scope:
            t_node = type(node)
            if t_node in node_expectation:

                feature_idx = feature_scope.index(node.scope[0])
                inverted = inverted_features[feature_idx]

                result = node_expectation[t_node](node, evidence, inverted=inverted).item()
                
                print("exp - leaf - node_exp:", result)
                
                if node.id in list(node_status.keys()):
                    node_status[node.id].append(result)
                else:
                    node_status[node.id] = []
                    node_status[node.id].append(result)
               
                return result, node_status
            else:
                raise Exception('Node type unknown: ' + str(t_node))
        result = node_likelihoods[type(node)](node, evidence).item()
        print("exp - leaf - node_likelihoods:", result)
        
        if node.id in list(node_status.keys()):
            node_status[node.id].append(result)
        else:
            node_status[node.id] = []
            node_status[node.id].append(result)

        return result, node_status