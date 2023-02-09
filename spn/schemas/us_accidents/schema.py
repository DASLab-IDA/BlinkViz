from ensemble_compilation.graph_representation import SchemaGraph, Table

# 原始数据4232541
def gen_us_accident_schema(csv_path):
    schema = SchemaGraph()

    # tables
    schema.add_table(Table('us_accidents',
                           attributes=['ID', 'Source', 'TMC','Severity','Start_Time','End_Time','Start_Lat','Start_Lng','End_Lat','End_Lng','Distance','Description','Number','Street','Side','City','County','State','Zipcode','Country','Timezone','Airport_Code','Weather_Timestamp','Temperature','Wind_Chill','Humidity','Pressure','Visibility','Wind_Direction','Wind_Speed','Precipitation','Weather_Conditio', 'Amenity','Bump','Crossing','Give_Way','Junction','No_Exit','Railway','Roundabout','Station','Stop','Traffic_Calming','Traffic_Signal','Turning_Loop','Sunrise_Sunset','Civil_Twilight','Nautical_Twilight', 'Astronomical_Twilight'],
                           csv_file_location=csv_path.format('dataset_sampled'),
                           table_size=10906860, primary_key=['ID'], sample_rate=0.1,
                           fd_list=[]
                           ))
 
    
    schema.table_dictionary['us_accidents'].sample_rate = 0.01
    schema.table_dictionary['us_accidents'].table_size = 4232541

    return schema