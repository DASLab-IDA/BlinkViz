from ensemble_compilation.graph_representation import SchemaGraph, Table

def gen_nyctaxi_schema(csv_path):
    schema = SchemaGraph()

    # tables
    schema.add_table(Table('yellowcab',
                           attributes=['VendorID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime', 'passenger_count', 'trip_distance',
                                       'RatecodeID', 'store_and_fwd_flag', 'PULocationID', 'DOLocationID', 'payment_type',
                                       'fare_amount', 'extra', 'mta_tax', 'tip_amount', 'tolls_amount', 'improvement_surcharge', 'total_amount','congestion_surcharge'],
                           csv_file_location=csv_path.format('dataset_sampled'),
                           table_size=6405008, primary_key=['VendorID'], sample_rate=0.01,
                           fd_list=[]
                           ))

    schema.table_dictionary['yellowcab'].sample_rate = 0.01
    schema.table_dictionary['yellowcab'].table_size = 6405008

    return schema