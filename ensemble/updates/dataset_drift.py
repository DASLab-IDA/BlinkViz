def dataset_drift_detection_naive(attrs, attrs_types, db_conn, k, table_name):
    # 对每次插入数据前后都计算一遍各个属性的分布情况（categorical求group by，用K-S检验，numerical用均值和方差）
    pass