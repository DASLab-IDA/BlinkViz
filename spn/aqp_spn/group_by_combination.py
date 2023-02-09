import copy
import logging

import numpy as np
from spn.algorithms.Inference import likelihood
from spn.structure.Base import get_nodes_by_type, Leaf, Product, eval_spn_bottom_up, assign_ids, bfs

from rspn.algorithms.transform_structure import Prune
from rspn.algorithms.validity.validity import is_valid
from rspn.structure.base import Sum

logger = logging.getLogger(__name__)


def prod_group_by(node, children, data=None, dtype=np.float64):
    contains_probs = False
    contains_values = False
    contains_none_values = False
    contains_zero_prob = False
    group_by_scopes = []
    print("prod_group_by - node:", node)
    print("prod_group_by - children:", children)
    # Check if only probabilities contained
    for child in children:
        # value
        if isinstance(child, tuple):
            contains_values = True

            scope, values = child
            group_by_scopes += scope
            if values is None:
                contains_none_values = True
        # probability
        else:
            contains_probs = True
            if (child == 0).any():
                contains_zero_prob = True

    # Probability of subtree zero or no matching tuples
    if contains_zero_prob or contains_none_values:
        print("prod_group_by - return [None], None")
        #with open("/home/qym/blinkviz/deepdb/flights-benchmark/ensemble_learning/eval_tmp.txt","a+") as eval_file:
        #    eval_file.write("prod_group_by\n")
        #    eval_file.write(node.name+"\n")
        #    eval_file.write("[None], None\n")
        return [None], None
    # Cartesian product
    elif contains_values:
        result_values = None
        group_by_scopes.sort()
        for group_by_scope in group_by_scopes:
            matching_values = None
            matching_idx = None
            for child in children:
                if isinstance(child, tuple):
                    scope, values = child
                    if group_by_scope in scope:
                        matching_values = values
                        matching_idx = scope.index(group_by_scope)
                        break
            assert matching_values is not None, "Matching values should not be None."
            if result_values is None:
                result_values = [(matching_value[matching_idx],) for matching_value in matching_values]
            else:
                result_values = [result_value + (matching_value[matching_idx],) for result_value in result_values for
                                 matching_value in matching_values]
                # assert len(result_values) <= len(group_by_scopes)
        print("prod_group_by - group_by_scopes:", group_by_scopes)
        print("prod_group_by - result_values:", result_values)
        #with open("/home/qym/blinkviz/deepdb/flights-benchmark/ensemble_learning/eval_tmp.txt","a+") as eval_file:
        #    eval_file.write("prod_group_by\n")
        #    eval_file.write(node.name+"\n")
        #    eval_file.write(str(set(result_values))+"\n")
        return group_by_scopes, set(result_values)
    # Only probabilities, normal inference
    elif contains_probs:
        llchildren = np.concatenate(children, axis=1)
        result = np.nanprod(llchildren, axis=1).reshape(-1, 1)
        #print("prod_group_by - np.nanprod(llchildren, axis=1).reshape(-1, 1):", np.nanprod(llchildren, axis=1).reshape(-1, 1))
        #with open("/home/qym/blinkviz/deepdb/flights-benchmark/ensemble_learning/eval_tmp.txt","a+") as eval_file:
        #    eval_file.write("prod_group_by\n")
        #    eval_file.write(node.name+"\n")
        #    eval_file.write(str(result)+"\n")
        return result


def sum_group_by(node, children, data=None, dtype=np.float64):
    """
    Propagate expectations in sum node.

    :param node: sum node
    :param children: nodes below
    :param data:
    :param dtype:
    :return:
    """
    print("sum_group_by - node:", node)
    print("sum_group_by - children:", children)

    """
    new_children = []

    flag = True
    for i in children:
        #print("i:", i)
        if isinstance(i, tuple)==False:
            flag = False
            new_children.append(i)
        else:
            if i[1] is not None:
                new_children.append(i)
            
    print("new_children:", new_children)
    """
    flag = True
    for idx in range(len(children)):
        if not isinstance(children[idx], tuple):
            flag = False
    # either all tuples or
    if isinstance(children[0], tuple) and flag==True:
        result_values = None
        group_by_scope = [None]
        for scope, values in children:
            if values is not None:
                group_by_scope = scope
                if result_values is None:
                    result_values = values
                else:
                    result_values = result_values.union(values)
        #with open("/home/qym/blinkviz/deepdb/flights-benchmark/ensemble_learning/eval_tmp.txt","a+") as eval_file:
        #    eval_file.write("sum_group_by\n")
        #    eval_file.write(node.name+"\n")
        #    eval_file.write(str(result_values)+"\n")
        print("sum_group_by - group_by_scope:", group_by_scope)
        print("sum_group_by - result_values:", result_values)
        #if result_values is None:
        #    return ([], {})
        return group_by_scope, result_values

    # normal probability sum node code
    # print("children:", children)
    for child_idx in range(len(children)):
        if isinstance(children[child_idx], tuple):
            if children[child_idx][1] is None:
                children[child_idx] = np.array([[0.]])

    llchildren = np.concatenate(children, axis=1)
    relevant_children_idx = np.where(np.isnan(llchildren[0]) == False)[0]
    if len(relevant_children_idx) == 0:
        print("sum_group_by - np.array([np.nan]):", np.array([np.nan]))
        #with open("/home/qym/blinkviz/deepdb/flights-benchmark/ensemble_learning/eval_tmp.txt","a+") as eval_file:
        #    eval_file.write("sum_group_by\n")
        #    eval_file.write(node.name+"\n")
        #    eval_file.write("np.array([np.nan])\n")
        return np.array([np.nan])

    assert llchildren.dtype == dtype

    weights_normalizer = sum(node.weights[j] for j in relevant_children_idx)
    b = np.array(node.weights, dtype=dtype)[relevant_children_idx] / weights_normalizer

    result = np.dot(llchildren[:, relevant_children_idx], b).reshape(-1, 1)
    print("result:", result)
    #with open("/home/qym/blinkviz/deepdb/flights-benchmark/ensemble_learning/eval_tmp.txt","a+") as eval_file:
    #    eval_file.write("sum_group_by\n")
    #    eval_file.write(node.name+"\n")
    #    eval_file.write(str(result)+"\n")
    # print("size(result):",len(result))
    return result


def group_by_combinations(spn, ds_context, feature_scope, ranges, node_distinct_vals=None, node_likelihoods=None):
    """
    Computes the distinct value combinations for features given the range conditions.
    """
    #print("group_by_combination - feature scope:", feature_scope)
    #print("group_by_combination - ranges:", ranges)
    #print("group_by_combination - ds_context:", ds_context)
    #print("group_by_combination - node_distinct_vals:", node_distinct_vals)
    #print("group_by_combination - node_likelihoods:", node_likelihoods)

    evidence_scope = set([i for i, r in enumerate(ranges[0]) if r is not None])
    evidence = ranges

    # make feature scope sorted
    feature_scope_unsorted = copy.copy(feature_scope)
    feature_scope.sort()
    # add range conditions to feature scope (makes checking with bloom filters easier)
    feature_scope = list(set(feature_scope)
                         .union(evidence_scope.intersection(np.where(ds_context.no_unique_values <= 1200)[0])))
    feature_scope.sort()
    inverted_order = [feature_scope.index(scope) for scope in feature_scope_unsorted]

    assert not (len(evidence_scope) > 0 and evidence is None)

    relevant_scope = set()
    relevant_scope.update(evidence_scope)
    relevant_scope.update(feature_scope)
    print("group_by_combinations - relevant_scope:", relevant_scope)
    marg_spn = marginalize(spn, relevant_scope)

    """
    def printNode(node):
        if isinstance(node, Sum):
            print("bfs - sum node:", node)
            print("bfs - sum node.scope:", node.scope)
            print("bfs - sum node.weights:", node.weights)
            print("bfs - sum node.children:", node.children)
            print("bfs - sum node.cluster_centers:", node.cluster_centers)
            print("bfs - sum node.cardinality:", node.cardinality)
        elif isinstance(node, Product):
            print("bfs - product node:", node)
            print("bfs - product node.scope:", node.scope)
            print("bfs - product node.children:", node.children)
        else:
            print("bfs - leaf node:", node)
            print("bfs - leaf node.scope:", node.scope)

    print("group_by_combination - bfs start ========================")
    bfs(marg_spn, printNode)
    """

    def leaf_expectation(node, data, dtype=np.float64, **kwargs):
        
        if node.scope[0] in feature_scope:
            t_node = type(node)
            if t_node in node_distinct_vals:
                vals = node_distinct_vals[t_node](node, evidence)
                return vals
            else:
                raise Exception('Node type unknown: ' + str(t_node))

        result = likelihood(node, evidence, node_likelihood=node_likelihoods)
        print("leaf_expectation - node:", node)
        print("leaf_expectation - node.scope:", node.scope)
        #print("leaf_expectation - result:", result)
        #with open("/home/qym/blinkviz/deepdb/flights-benchmark/ensemble_learning/eval_tmp.txt","a+") as eval_file:
        #   eval_file.write("leaf_expectation\n")
        #    eval_file.write(node.name+"\n")
        #    eval_file.write(str(result)+"\n")
        return result

    node_expectations = {type(leaf): leaf_expectation for leaf in get_nodes_by_type(marg_spn, Leaf)}
    node_expectations.update({Sum: sum_group_by, Product: prod_group_by})
    # 自低向上遍历，node_expectations是evaluation function
    result = eval_spn_bottom_up(marg_spn, node_expectations, all_results={}, data=evidence, dtype=np.float64)
    #print("group_by_combination - result:", result)
    if feature_scope_unsorted == feature_scope:
        return result
    scope, grouped_tuples = result
    print("group_by_combination - scope:", scope)
    print("group_by_combination - grouped_tuples:", grouped_tuples)
    print("group_by_combination - feature_scope_unsorted:", feature_scope_unsorted)
    print("group_by_combination - inverted_order:",  inverted_order)
    if grouped_tuples is not None:
        return feature_scope_unsorted, set(
        [tuple(group_tuple[i] for i in inverted_order) for group_tuple in grouped_tuples])
    else:
        return feature_scope_unsorted, set()


def marginalize(node, keep, light=False):
    # keep must be a set of features that you want to keep
    # Loc.enter()
    keep = set(keep)

    #print("marginalize - node:", node)

    # Loc.p('keep:', keep)

    def marg_recursive(node):
        #print("marg_recursive - node:", node)
        # Loc.enter()
        new_node_scope = keep.intersection(set(node.scope))
        # print("marginalize - new_node_scope:", new_node_scope)
        # Loc.p("new_node_scope:", new_node_scope)
        if len(new_node_scope) == 0:
            # we are summing out this node
            # Loc.leave(None)
            return None

        if isinstance(node, Leaf):
            #print("marginalize - leaf node:", node)
            if len(node.scope) > 1:
                raise Exception("Leaf Node with |scope| > 1")
            # Loc.leave('Leaf.deepcopy()')
            if light:
                return node
            return copy.deepcopy(node)

        newNode = node.__class__()
        newNode.cardinality = node.cardinality

        if isinstance(node, Sum):
            newNode.weights.extend(node.weights)
            if not light:
                newNode.cluster_centers.extend(node.cluster_centers)

        #print("marginalize - node.children:", node.children)
        for c in node.children:
            new_c = marg_recursive(c)
            if new_c is None:
                continue
            newNode.children.append(new_c)

        newNode.scope.extend(new_node_scope)

        # Loc.leave()
        return newNode

    newNode = marg_recursive(node)

    if not light:
        assign_ids(newNode)
        newNode = Prune(newNode, check_cluster_centers=light)

        valid, err = is_valid(newNode, check_cluster_centers=light)
        assert valid, err
    # Loc.leave()
    return newNode