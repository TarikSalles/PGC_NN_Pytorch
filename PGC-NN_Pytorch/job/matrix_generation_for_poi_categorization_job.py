import numpy as np
import pandas as pd

from domain.matrix_generation_for_poi_categorization_domain import MatrixGenerationForPoiCategorizationDomain
from extractor.file_extractor import FileExtractor
from configuration.poi_categorization_configuration import PoICategorizationConfiguration
from configuration.matrix_generation_for_poi_categorization_configuration import MatrixGenerationForPoiCategorizationConfiguration
from loader.poi_categorization_loader import PoiCategorizationLoader
from pathlib import Path

class MatrixGenerationForPoiCategorizationJob():

    def __init__(self):
        self.file_extractor = FileExtractor()
        self.matrix_generation_for_poi_categorization_domain = MatrixGenerationForPoiCategorizationDomain("gowalla")
        self.poi_categorization_loader = PoiCategorizationLoader()
        self.poi_categorization_configuration = PoICategorizationConfiguration()

    def start(self):
        users_checkin_filename = "gowalla/checkins.csv"
        adjacency_matrix_base_filename = "adjacency_matrix"
        features_matrix_base_filename = "features_matrix"
        distance_matrix_base_filename = "distance_matrix"
        duration_matrix_base_filename = "duration_matrix"
        dataset_name = "gowalla"
        categories_type = "7_categories"
        country = "United States"
        print("\nDataset: ", dataset_name)

        convert_country = {'Brazil': 'BR', 'BR': 'BR', 'United States': 'US'}
        hour_file = "48_"

        userid_column = MatrixGenerationForPoiCategorizationConfiguration.DATASET_COLUMNS.get_value()[dataset_name]['userid_column']
        category_column = MatrixGenerationForPoiCategorizationConfiguration.DATASET_COLUMNS.get_value()[dataset_name]['category_column']
        category_name_column = MatrixGenerationForPoiCategorizationConfiguration.DATASET_COLUMNS.get_value()[dataset_name]['category_name_column']
        locationid_column = MatrixGenerationForPoiCategorizationConfiguration.DATASET_COLUMNS.get_value()[dataset_name]['locationid_column']
        datetime_column = MatrixGenerationForPoiCategorizationConfiguration.DATASET_COLUMNS.get_value()[dataset_name]['datetime_column']
        latitude_column = MatrixGenerationForPoiCategorizationConfiguration.DATASET_COLUMNS.get_value()[dataset_name]['latitude_column']
        longitude_column = MatrixGenerationForPoiCategorizationConfiguration.DATASET_COLUMNS.get_value()[dataset_name]['longitude_column']
        country_column = MatrixGenerationForPoiCategorizationConfiguration.DATASET_COLUMNS.get_value()[dataset_name]['country_column']
        category_to_int = self.poi_categorization_configuration.CATEGORIES_TO_INT[dataset_name][categories_type]

        dtypes_columns = {userid_column: int, category_column: 'Int16', category_name_column: 'category',
                          locationid_column: 'category', datetime_column: 'category', latitude_column: 'float64',
                          longitude_column: 'float64'}

        print(dtypes_columns)

        users_checkin = self.file_extractor.read_csv(users_checkin_filename, dtypes_columns).query(country_column + " == '"+country+"'")
        if category_column == category_name_column:
            categories = users_checkin[category_name_column].tolist()
            categories_int = []
            for i in range(len(categories)):
                if categories[i] == 'Other':
                    categories_int.append(-1)
                else:
                    categories_int.append(category_to_int[categories[i]])

            category_column = category_column + "_id"
            users_checkin[category_column] = np.array(categories_int)
            print("\n\n", users_checkin[category_column], "\n\n", users_checkin[category_column].unique() )

        print("\n----- verificação -----")
        print("\nPais: ", users_checkin[country_column].unique().tolist())

        # data

        users_checkin[datetime_column] = pd.to_datetime(users_checkin[datetime_column], infer_datetime_format=True)
        users_checkin[category_column] = users_checkin[category_column].astype('int')

        """
        Generate matrixes for each user 
        """
        folder = 'gowalla/'
        self.folder_generation(folder)
        country = convert_country[country]
        adjacency_matrix_filename = folder + adjacency_matrix_base_filename + "_not_directed_" + hour_file + categories_type + "_" + country + ".csv"
        adjacency_weekday_matrix_filename = folder + adjacency_matrix_base_filename + "_weekday_not_directed_"+hour_file+categories_type+"_"+country+".csv"
        adjacency_weekend_matrix_filename = folder+adjacency_matrix_base_filename + "_weekend_not_directed_"+hour_file+categories_type+"_"+country+".csv"
        temporal_matrix_filename = folder+features_matrix_base_filename + "_not_directed_"+hour_file+categories_type+"_"+country+".csv"
        temporal_weekday_matrix_filename = folder+features_matrix_base_filename + "_weekday_not_directed_"+hour_file+categories_type+"_"+country+".csv"
        temporal_weekend_matrix_filename = folder+features_matrix_base_filename + "_weekend_not_directed_"+hour_file+categories_type+"_"+country+".csv"
        distance_matrix_filename = folder + distance_matrix_base_filename + "_not_directed_" + hour_file + categories_type + "_" + country + ".csv"
        duration_matrix_filename = folder + duration_matrix_base_filename + "_not_directed_" + hour_file + categories_type + "_" + country + ".csv"
        location_locaion_pmi_matrix_filename = folder + "location_location_pmi_matrix_" + categories_type + "_" + country + ".npz"
        location_time_pmi_matrix_filename = folder + "location_time_pmi_matrix_" + categories_type + "_" + country + ".csv"
        int_to_locationid_filename = folder + "int_to_locationid_" + categories_type + "_" + country + ".csv"
        
        self.matrix_generation_for_poi_categorization_domain\
            .generate_pattern_matrices(users_checkin,
                                        adjacency_matrix_filename,
                                        adjacency_weekday_matrix_filename,
                                        adjacency_weekend_matrix_filename,
                                        temporal_matrix_filename,
                                        temporal_weekday_matrix_filename,
                                        temporal_weekend_matrix_filename,
                                        distance_matrix_filename,
                                        duration_matrix_filename,
                                        location_locaion_pmi_matrix_filename,
                                        location_time_pmi_matrix_filename,
                                        int_to_locationid_filename,
                                        userid_column,
                                        category_column,
                                        locationid_column,
                                        latitude_column,
                                        longitude_column,
                                        datetime_column)

    def folder_generation(self, folder):
        Path(folder).mkdir(parents=True, exist_ok=True)