import numpy as np
import pandas as pd
import tensorflow as tf

from domain.poi_categorization_domain import PoiCategorizationDomain
from extractor.file_extractor import FileExtractor
from configuration.poi_categorization_configuration import PoICategorizationConfiguration
from loader.poi_categorization_loader import PoiCategorizationLoader

class PoiCategorizationJob:

    def __init__(self):
        self.file_extractor = FileExtractor()
        self.poi_categorization_domain = PoiCategorizationDomain("gowalla")
        self.poi_categorization_loader = PoiCategorizationLoader()
        self.poi_categorization_configuration = PoICategorizationConfiguration()

    def start(self):
        base_dir = "gowalla/"
        adjacency_matrix_filename = "gowalla/adjacency_matrix_not_directed_48_7_categories_US.csv"
        adjacency_matrix_week_filename = "gowalla/adjacency_matrix_weekday_not_directed_48_7_categories_US.csv"
        adjacency_matrix_weekend_filename = "gowalla/adjacency_matrix_weekend_not_directed_48_7_categories_US.csv"
        graph_type = "not_directed"
        temporal_matrix_filename = "gowalla/features_matrix_not_directed_48_7_categories_US.csv"
        temporal_matrix_week_filename = "gowalla/features_matrix_weekday_not_directed_48_7_categories_US.csv"
        temporal_matrix_weekend_filename = "gowalla/features_matrix_weekend_not_directed_48_7_categories_US.csv"
        distance_matrix_filename = "gowalla/distance_matrix_not_directed_48_7_categories_US.csv"
        duration_matrix_filename = "gowalla/duration_matrix_not_directed_48_7_categories_US.csv"
        dataset_name = "gowalla"
        categories_type = "7_categories"
        location_location_filename = "gowalla/location_location_pmi_matrix_7_categories_US.npz"
        location_time_filename = "gowalla/location_time_pmi_matrix_7_categories_US.csv"
        int_to_locationid_filename = "gowalla/int_to_locationid_7_categories_US.csv"
        country = "US"
        state = "Alabama"
        version = "normal"

        print("\nDataset: ", dataset_name)

        max_size_matrices = self.poi_categorization_configuration.MAX_SIZE_MATRICES[1]
        max_size_paths = self.poi_categorization_configuration.MINIMUM_RECORDS[1]
        n_splits = self.poi_categorization_configuration.N_SPLITS[1]
        n_replications = self.poi_categorization_configuration.N_REPLICATIONS[1]
        epochs = self.poi_categorization_configuration.EPOCHS[1][country]
        output_base_dir = self.poi_categorization_configuration.OUTPUT_DIR[1]
        dataset_type_dir = self.poi_categorization_configuration.DATASET_TYPE[1][dataset_name]
        category_type_dir = self.poi_categorization_configuration.CATEGORY_TYPE[1][categories_type]
        int_to_category = self.poi_categorization_configuration.INT_TO_CATEGORIES[1][dataset_name][categories_type]
        graph_type_dir = self.poi_categorization_configuration.GRAPH_TYPE[1][graph_type]
        country_dir = self.poi_categorization_configuration.COUNTRY[1][country]
        state_dir = self.poi_categorization_configuration.STATE[1][state]
        version_dir = self.poi_categorization_configuration.VERSION[1][version]

        output_dir = (
            self.poi_categorization_configuration
            .output_dir(
                output_base_dir=output_base_dir,
                base="base/",
                graph_type=graph_type_dir,
                dataset_type=dataset_type_dir,
                country=country_dir,
                category_type=category_type_dir,
                version=version_dir,
                state_dir=state_dir,
                max_time_between_records_dir=""
            )
        )


        base_report = self.poi_categorization_configuration.REPORT_MODEL[1][categories_type]
        base_dir = "gowalla/"

        # normal matrices
        (
            adjacency_df, temporal_df, distance_df, duration_df
        ) = (
            self.poi_categorization_domain.
            read_matrix(
                adjacency_matrix_filename, 
                temporal_matrix_filename, 
                distance_matrix_filename=distance_matrix_filename, 
                duration_matrix_filename=duration_matrix_filename
            )
        )

        # week matrices
        adjacency_week_df, temporal_week_df = (
            self.poi_categorization_domain.
            read_matrix(
                adjacency_matrix_week_filename, 
                temporal_matrix_week_filename
            )
        )
        # weekend matrices
        adjacency_weekend_df, temporal_weekend_df = (
            self.poi_categorization_domain.
            read_matrix(
                adjacency_matrix_weekend_filename, 
                temporal_matrix_weekend_filename
            )
        )

        print("\nVerificação de matrizes\n")
        self.matrices_verification([
            adjacency_df, 
            temporal_df, 
            adjacency_week_df, 
            temporal_week_df,
            adjacency_weekend_df, 
            temporal_weekend_df, 
            distance_df, 
            duration_df
        ])

        location_location = self.file_extractor.read_npz(location_location_filename)
        location_time = self.file_extractor.read_csv( location_time_filename)
        int_to_locationid = self.file_extractor.read_csv( int_to_locationid_filename)

        inputs = {
            'all_week': {
                'adjacency': adjacency_df, 
                'temporal': temporal_df, 
                'distance': distance_df, 
                'duration': duration_df,
                'location_location': location_location, 
                'location_time': location_time, 
                'int_to_locationid': int_to_locationid
            },
            'week': {
                'adjacency': adjacency_week_df, 
                'temporal': temporal_week_df
            },
            'weekend': {
                'adjacency': adjacency_weekend_df, 
                'temporal': temporal_weekend_df
            }
        }

        print("\nPreprocessing\n")
        (
            users_categories, 
            adjacency_df, 
            temporal_df, 
            distance_df, 
            duration_df, 
            adjacency_week_df, 
            temporal_week_df,
            adjacency_weekend_df, 
            temporal_weekend_df, 
            location_time_df, 
            location_location_df, 
            selected_users
        ) = (
            self.poi_categorization_domain
            .poi_gnn_adjacency_preprocessing(
                inputs,
                max_size_matrices,
                dataset_name
            )
        )


        selected_users = pd.DataFrame({'selected_users': selected_users})

        self.matrices_verification([
            adjacency_df, 
            temporal_df, 
            adjacency_week_df, 
            temporal_week_df,
            adjacency_weekend_df, 
            temporal_weekend_df, 
            distance_df
        ])
        
        inputs = {
            'all_week': {
                'adjacency': adjacency_df, 
                'temporal': temporal_df, 
                'location_time': location_time_df,
                'location_location': location_location_df, 
                'categories': users_categories,
                'distance': distance_df, 
                'duration': duration_df
            },
            'week': {
                'adjacency': adjacency_week_df, 
                'temporal': temporal_week_df,
                'categories': users_categories
            },
            'weekend': {
                'adjacency': adjacency_weekend_df, 
                'temporal': temporal_weekend_df,
                'categories': users_categories
            }
        }

        usuarios = len(adjacency_df)

        folds, class_weight = (
            self.poi_categorization_domain.
            k_fold_split_train_test(
                max_size_matrices,
                inputs,
                n_splits,
                'all_week'
            )
        )

        folds_week, class_weight_week = (
            self.poi_categorization_domain.
            k_fold_split_train_test(
                max_size_matrices,
                inputs,
                n_splits,
                'week'
            )
        )

        folds_weekend, class_weight_weekend = (
            self.poi_categorization_domain.
            k_fold_split_train_test(
                max_size_matrices,
                inputs,
                n_splits,
                'weekend'
            )
        )

        print("\nclass weight: ", class_weight)
        inputs_folds = {
            'all_week': {
                'folds': folds, 
                'class_weight': class_weight
            },
            'week': {
                'folds': folds_week, 
                'class_weight': class_weight_week
            },
            'weekend': {
                'folds': folds_weekend, 
                'class_weight': class_weight_weekend
            }
        }

        print("\nTreino\n")
        folds_histories, base_report, model = (
            self.poi_categorization_domain.
            k_fold_with_replication_train_and_evaluate_model(
                inputs_folds,
                n_replications,
                max_size_matrices,
                max_size_paths,
                base_report,
                epochs,
                class_weight,
                country,
                version,
                output_dir
            )
        )

        selected_users.to_csv(output_dir + "selected_users.csv", index=False)
        print("\nbase: ", base_dir)
        base_report = self.poi_categorization_domain.preprocess_report(base_report, int_to_category)
        self.poi_categorization_loader.plot_history_metrics(folds_histories, base_report, output_dir)
        self.poi_categorization_loader.save_report_to_csv(output_dir, base_report, n_splits, n_replications, usuarios)
        self.poi_categorization_loader.save_model_and_weights(model, output_dir, n_splits, n_replications)
        print("\nUsuarios processados: ", usuarios)

    def matrices_verification(self, df_list):

        for i in range(1, len(df_list)):
            if not(len(df_list[i-1]) == len(df_list[i])):
                print("\nMatrizes com tamanhos diferentes\n")
                raise