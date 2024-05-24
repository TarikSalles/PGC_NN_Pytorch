import json

import numpy as np
import pandas as pd
import spektral as sk
import spektral.layers
import torch
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import utils as np_utils
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset
from torch_geometric.data import DataLoader
from torch_geometric.graphgym import optim

from extractor.file_extractor import FileExtractor
from globals import Configurations
from loader.file_loader import FileLoader
from loader.poi_categorization_loader import PoiCategorizationLoader
from model.gnn_base_model_for_transfer_learning import GNNUS_BaseModel
from utils.nn_preprocessing import top_k_rows, split_graph, \
    top_k_rows_category_user_tracking, prepare_pyg_batch, one_hot_decoding_predicted


class PoiCategorizationDomain:

    def __init__(self, dataset_name):
        self.file_loader = FileLoader()
        self.file_extractor = FileExtractor()
        self.poi_categorization_loader = PoiCategorizationLoader()
        self.dataset_name = dataset_name

    def read_matrix(self, adjacency_matrix_filename, temporal_matrix_filename, distance_matrix_filename=None,
                    duration_matrix_filename=None):

        adjacency_df = self.file_extractor.read_csv(adjacency_matrix_filename).drop_duplicates(subset=['user_id'])
        temporal_matrix_df = self.file_extractor.read_csv(temporal_matrix_filename).drop_duplicates(subset=['user_id'])

        if (
                distance_matrix_filename is not None and
                duration_matrix_filename is not None
        ):
            distance_matrix_df = self.file_extractor.read_csv(distance_matrix_filename).drop_duplicates(
                subset=['user_id'])
            duration_matrix_df = self.file_extractor.read_csv(duration_matrix_filename).drop_duplicates(
                subset=['user_id'])
            if adjacency_df['user_id'].tolist() != temporal_matrix_df['user_id'].tolist():
                print("\nMATRIZES DIFERENTES\n")
                raise

            return (
                adjacency_df,
                temporal_matrix_df,
                distance_matrix_df,
                duration_matrix_df
            )

        else:
            if adjacency_df['user_id'].tolist() != temporal_matrix_df['user_id'].tolist():
                print("\nMATRIZES DIFERENTES\n")
                raise
            return adjacency_df, temporal_matrix_df

    def _poi_gnn_resize_adjacency_and_category_matrices(
            self,
            user_matrix,
            user_matrix_week,
            user_matrix_weekend,
            user_category,
            max_size_matrices,
            dataset_name
    ):

        more_matrices = 1
        k_original = max_size_matrices
        size = user_matrix.shape[0]
        if size < k_original:
            k = user_matrix.shape[0]
        else:
            k = int(np.floor(size / k_original) * k_original)
        # select the k rows that have the highest sum
        if dataset_name == "user_tracking":
            idx = top_k_rows_category_user_tracking(user_matrix, k, user_category)
        else:
            idx = top_k_rows(user_matrix, k)

        not_used_ids = []
        for i in range(len(idx)):

            if i not in idx:
                not_used_ids.append(i)

        if len(not_used_ids) > 0 or size < max_size_matrices:

            add_more = max_size_matrices - len(not_used_ids)

            count = 0
            i = 0
            for i in idx:

                if count < add_more:

                    not_used_ids.append(i)
                    count += 1

                else:

                    break

        idx = np.array(idx.tolist() + not_used_ids)

        user_matrix = user_matrix[idx[:, None], idx]
        user_matrix_week = user_matrix_week[idx[:, None], idx]
        user_matrix_weekend = user_matrix_weekend[idx[:, None], idx]
        user_category = user_category[idx]

        if k > k_original or len(not_used_ids) > 0:
            k_split = int(np.floor(size / k_original))
            if len(not_used_ids) > 0:
                k_split += 1
            user_matrix = split_graph(user_matrix, k_original, k_split)
            user_matrix_week = split_graph(user_matrix_week, k_original, k_split)
            user_matrix_weekend = split_graph(user_matrix_weekend, k_original, k_split)
            user_category = split_graph(user_category, k_original, k_split)
            idx = split_graph(idx, k_original, k_split)
            more_matrices = k_split

            return (
                np.array(user_matrix),
                np.array(user_matrix_week),
                np.array(user_matrix_weekend),
                np.array(user_category),
                np.array(idx),
                more_matrices
            )

        return (
            np.array([user_matrix]),
            np.array([user_matrix_week]),
            np.array([user_matrix_weekend]),
            np.array([user_category]),
            np.array([idx]),
            more_matrices
        )

    def _filter_pmi_matrix(self, location_time, location_location, locationid_to_int, visited_location_ids):

        idx = np.array([locationid_to_int[visited_location_ids[i]] for i in range(len(visited_location_ids))])

        location_time = location_time[idx]
        location_location = location_location[idx[:, None], idx].toarray()
        location_location = sk.layers.GCNConv.preprocess(location_location)

        return location_time, location_location

    def poi_gnn_adjacency_preprocessing(self,
                                        inputs,
                                        max_size_matrices,
                                        dataset_name,
                                        ):
        matrices_list = []
        temporal_matrices_list = []
        distance_matrices_list = []
        duration_matrices_list = []
        # week
        matrices_week_list = []
        temporal_matrices_week_list = []
        # weekend
        matrices_weekend_list = []
        temporal_matrices_weekend_list = []
        # location time
        location_time_list = []
        location_location_list = []

        users_categories = []
        maior = -10

        matrix_df = inputs['all_week']['adjacency']
        ids = matrix_df['user_id'].tolist()
        matrix_df = matrix_df['matrices'].tolist()
        category_df = inputs['all_week']['adjacency']['category'].tolist()
        temporal_df = inputs['all_week']['temporal']['matrices'].tolist()
        distance_df = inputs['all_week']['distance']['matrices'].tolist()
        duration_df = inputs['all_week']['duration']['matrices'].tolist()
        visited_location_ids = inputs['all_week']['adjacency']['visited_location_ids'].tolist()
        location_location_df = inputs['all_week']['location_location']
        location_time_df = inputs['all_week']['location_time'].to_numpy()
        locationid_to_int = inputs['all_week']['int_to_locationid']
        locationid_to_int_ids = locationid_to_int['locationid'].tolist()
        locationid_to_int_ints = locationid_to_int['int'].tolist()
        locationid_to_int = {locationid_to_int_ids[i]: locationid_to_int_ints[i] for i in
                             range(len(locationid_to_int_ints))}
        # week
        matrix_week_df = inputs['week']['adjacency']['matrices'].tolist()
        temporal_week_df = inputs['week']['temporal']['matrices'].tolist()
        # weekend
        matrix_weekend_df = inputs['weekend']['adjacency']['matrices'].tolist()
        temporal_weekend_df = inputs['weekend']['temporal']['matrices'].tolist()

        selected_visited_locations = []

        if len(ids) != len(matrix_df):
            print("\nERRO TAMANHO DA MATRIZ\n")
            exit()

        selected_users = []
        remove = 0
        for i in range(len(ids)):

            number_of_matrices = 1
            user_id = ids[i]

            user_matrices = matrix_df[i]
            user_category = category_df[i]
            user_matrices = json.loads(user_matrices)
            user_matrices = np.array(user_matrices)
            user_category = json.loads(user_category)
            user_category = np.array(user_category)
            # week
            user_matrices_week = matrix_week_df[i]
            user_matrices_week = json.loads(user_matrices_week)
            user_matrices_week = np.array(user_matrices_week)
            # weekend
            user_matrices_weekend = matrix_weekend_df[i]
            user_matrices_weekend = json.loads(user_matrices_weekend)
            user_matrices_weekend = np.array(user_matrices_weekend)
            # user visited
            user_visited = visited_location_ids[i]
            user_visited = json.loads(user_visited)
            user_visited = np.array(user_visited)
            size = user_matrices.shape[0]
            if size > maior:
                maior = size

            # matrices get new size, equal for everyone
            (
                user_matrices,
                user_matrices_week,
                user_matrices_weekend,
                user_category,
                idxs,
                number_of_matrices
            ) = (
                self._poi_gnn_resize_adjacency_and_category_matrices(
                    user_matrices,
                    user_matrices_week,
                    user_matrices_weekend,
                    user_category,
                    max_size_matrices,
                    dataset_name
                )
            )

            """feature"""
            user_temporal_matrices = temporal_df[i]
            user_temporal_matrices = json.loads(user_temporal_matrices)
            user_temporal_matrices = np.array(user_temporal_matrices)
            # week
            user_temporal_matrices_week = temporal_week_df[i]
            user_temporal_matrices_week = json.loads(user_temporal_matrices_week)
            user_temporal_matrices_week = np.array(user_temporal_matrices_week)
            # weekend
            user_temporal_matrices_weekend = temporal_weekend_df[i]
            user_temporal_matrices_weekend = json.loads(user_temporal_matrices_weekend)
            user_temporal_matrices_weekend = np.array(user_temporal_matrices_weekend)
            """distance"""
            user_distance_matrix = distance_df[i]
            user_distance_matrix = json.loads(user_distance_matrix)
            user_distance_matrix = np.array(user_distance_matrix)
            """duration"""
            user_duration_matrix = duration_df[i]
            user_duration_matrix = json.loads(user_duration_matrix)
            user_duration_matrix = np.array(user_duration_matrix)

            for i in range(number_of_matrices):

                idx = idxs[i]
                matrices_list.append(sk.layers.ARMAConv.preprocess(user_matrices[i]))
                matrices_week_list.append(sk.layers.ARMAConv.preprocess(user_matrices_week[i]))
                matrices_weekend_list.append(sk.layers.ARMAConv.preprocess(user_matrices_weekend[i]))

                user_temporal_matrix = user_temporal_matrices[idx]
                temporal_matrices_list.append(self._min_max_normalize(user_temporal_matrix))
                user_temporal_matrix_week = user_temporal_matrices_week[idx]
                temporal_matrices_week_list.append(self._min_max_normalize(user_temporal_matrix_week))
                user_temporal_matrix_weekend = user_temporal_matrices_weekend[idx]
                temporal_matrices_weekend_list.append(self._min_max_normalize(user_temporal_matrix_weekend))
                distance_matrices_list.append(user_distance_matrix[idx[:, None], idx])
                duration_matrices_list.append(user_duration_matrix[idx[:, None], idx])
                users_categories.append(user_category[i])
                # location time
                user_location_time, user_location_location = (
                    self._filter_pmi_matrix(
                        location_time_df,
                        location_location_df,
                        locationid_to_int,
                        user_visited[idx]
                    )
                )
                user_location_time = self._min_max_normalize(user_location_time)
                location_time_list.append(user_location_time)
                user_location_location = spektral.layers.ARMAConv.preprocess(user_location_location)
                location_location_list.append(user_location_location)

                for j in user_visited[idx]:
                    selected_visited_locations.append(j)
                    selected_users.append(user_id)

        print("\nQuantidade de usuários", len(ids), " Quantidade removidos: ", remove, "\n")
        self.features_num_columns = temporal_matrices_list[-1].shape[1]
        matrices_list = np.array(matrices_list)
        location_time_list = np.array(location_time_list)
        location_location_list = np.array(location_location_list)
        temporal_matrices_list = np.array(temporal_matrices_list)
        users_categories = np.array(users_categories)

        distance_matrices_list = np.array(distance_matrices_list)
        duration_matrices_list = np.array(duration_matrices_list)

        # week
        matrices_week_list = np.array(matrices_week_list)
        temporal_matrices_week_list = np.array(temporal_matrices_week_list)

        # weekend
        matrices_weekend_list = np.array(matrices_weekend_list)
        temporal_matrices_weekend_list = np.array(temporal_matrices_weekend_list)
        temporal_matrices_week_list = np.array(temporal_matrices_week_list)

        return (
            users_categories,
            matrices_list,
            temporal_matrices_list,
            distance_matrices_list,
            duration_matrices_list,
            matrices_week_list,
            temporal_matrices_week_list,
            matrices_weekend_list,
            temporal_matrices_weekend_list,
            location_time_list,
            location_location_list,
            selected_users
        )

    def k_fold_split_train_test(self,
                                k,
                                inputs,
                                n_splits,
                                week_type,
                                model_name='poi_gnn'):

        adjacency_list = inputs[week_type]['adjacency']
        temporal_list = inputs[week_type]['temporal']
        user_categories = inputs[week_type]['categories']
        if model_name == "poi_gnn" and week_type == 'all_week':
            distance_list = inputs[week_type]['distance']
            duration_list = inputs[week_type]['duration']
            location_time = inputs[week_type]['location_time']
            location_location_list = inputs[week_type]['location_location']
        else:
            distance_list = []
            duration_list = []
            location_time = []
            location_location_list = []
        skip = False
        if n_splits == 1:
            skip = True
            n_splits = 2
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)

        folds = []
        classes_weights = []
        for train_indexes, test_indexes in kf.split(adjacency_list):

            fold, class_weight = (
                self._split_train_test(
                    k,
                    model_name,
                    adjacency_list,
                    user_categories,
                    temporal_list,
                    location_time,
                    location_location_list,
                    distance_list,
                    duration_list,
                    train_indexes,
                    test_indexes
                )
            )

            folds.append(fold)
            classes_weights.append(class_weight)
            if skip:
                break

        return folds, classes_weights

    def _split_train_test(self,
                          k,
                          model_name,
                          adjacency_list,
                          user_categories,
                          temporal_list,
                          location_time_list,
                          location_location_list,
                          distance_list,
                          duration_list,
                          train_indexes,
                          test_indexes):

        size = adjacency_list.shape[0]
        # 'average', 'cv', 'median', 'radius', 'label'
        adjacency_list_train = adjacency_list[train_indexes]
        user_categories_train = user_categories[train_indexes]

        temporal_list_train = temporal_list[train_indexes]

        if len(distance_list) > 0:
            distance_list_train = distance_list[train_indexes]
            duration_list_train = duration_list[train_indexes]
            location_time_list_train = location_time_list[train_indexes]
            location_location_list_train = location_location_list[train_indexes]
        else:
            distance_list_train = []
            duration_list_train = []
            location_time_list_train = []
            location_location_list_train = []

        adjacency_list_test = adjacency_list[test_indexes]
        user_categories_test = user_categories[test_indexes]
        temporal_list_test = temporal_list[test_indexes]

        if len(distance_list) > 0:
            distance_list_test = distance_list[test_indexes]
            duration_list_test = duration_list[test_indexes]
            location_time_list_test = location_time_list[test_indexes]
            location_location_list_test = location_location_list[test_indexes]
        else:
            distance_list_test = []
            duration_list_test = []
            location_time_list_test = []
            location_location_list_test = []

        flatten_train_category = []
        for categories_list in user_categories_train:
            flatten_train_category += categories_list.tolist()
        flatten_train_category = pd.Series(flatten_train_category, name='category')
        flatten_train_category = flatten_train_category.astype('object')
        train_categories_freq = {e: 0 for e in flatten_train_category.unique().tolist()}
        for i in range(flatten_train_category.shape[0]):
            train_categories_freq[flatten_train_category.iloc[i]] += 1
        n = sum(train_categories_freq.values())

        total_support = 0
        for e in train_categories_freq:
            total_support += train_categories_freq[e]

        total_support_inverse = 0
        for e in train_categories_freq:
            total_support_inverse += total_support - train_categories_freq[e]

        for e in train_categories_freq:
            train_categories_freq[e] = (total_support - train_categories_freq[e]) / total_support_inverse

        class_weight = list(train_categories_freq.values())
        user_categories_train = np.array([[e for e in row] for row in user_categories_train])
        user_categories_test = np.array([[e for e in row] for row in user_categories_test])

        if len(distance_list) > 0:
            return (
                (
                    adjacency_list_train,
                    user_categories_train,
                    temporal_list_train,
                    distance_list_train,
                    duration_list_train,
                    location_time_list_train,
                    location_location_list_train,
                    adjacency_list_test,
                    user_categories_test,
                    temporal_list_test,
                    distance_list_test,
                    duration_list_test,
                    location_time_list_test,
                    location_location_list_test,
                ),
                class_weight
            )
        else:
            return (
                (
                    adjacency_list_train,
                    user_categories_train,
                    temporal_list_train,
                    adjacency_list_test,
                    user_categories_test,
                    temporal_list_test
                ),
                class_weight
            )

    def k_fold_with_replication_train_and_evaluate_model(self,
                                                         inputs_folds,
                                                         n_replications,
                                                         max_size_matrices,
                                                         max_size_sequence,
                                                         base_report,
                                                         epochs,
                                                         class_weight,
                                                         country,
                                                         version,
                                                         output_dir):

        folds_histories = []

        folds_reports = []
        models = []
        accuracies = []
        seed = 0
        for i in range(len(inputs_folds['all_week']['folds'])):

            fold = inputs_folds['all_week']['folds'][i]
            fold_week = inputs_folds['week']['folds'][i]
            fold_weekend = inputs_folds['weekend']['folds'][i]
            class_weight = inputs_folds['all_week']['class_weight'][i]
            class_weight_week = inputs_folds['week']['class_weight'][i]
            class_weight_weekend = inputs_folds['weekend']['class_weight'][i]
            histories = []
            reports = []

            for _ in range(n_replications):
                history, report, model, accuracy = self.train_and_evaluate_model(i,
                                                                                 fold,
                                                                                 fold_week,
                                                                                 fold_weekend,
                                                                                 class_weight,
                                                                                 class_weight_week,
                                                                                 class_weight_weekend,
                                                                                 max_size_matrices,
                                                                                 max_size_sequence,
                                                                                 epochs,
                                                                                 seed,
                                                                                 country,
                                                                                 output_dir,
                                                                                 version)

                seed += 1

                base_report = self._add_location_report(base_report, report)
                histories.append(history)
                reports.append(report)
                models.append(model)
                accuracies.append(accuracy)
            folds_histories.append(histories)
            folds_reports.append(reports)
        best_model = self._find_best_model(models, accuracies)

        return folds_histories, base_report, best_model

    def train_and_evaluate_model(self, fold_number, fold, fold_week, fold_weekend,
                                 class_weight, class_weight_week, class_weight_weekend,
                                 max_size_matrices, max_size_sequence, epochs, seed,
                                 country,
                                 output_dir,
                                 version="normal",
                                 model=None
                                 ):
        (adjacency_train,
         y_train,
         temporal_train,
         distance_train,
         duration_train,
         location_time_train,
         location_location_train,
         adjacency_test,
         y_test,
         temporal_test,
         distance_test,
         duration_test,
         location_time_test,
         location_location_test) = fold

        (adjacency_week_train,
         y_train_week,
         temporal_train_week,
         adjacency_test_week,
         y_test_week,
         temporal_test_week) = fold_week

        (adjacency_train_weekend,
         y_train_weekend,
         temporal_train_weekend,
         adjacency_test_weekend,
         y_test_weekend,
         temporal_test_weekend) = fold_weekend

        max_total = 0
        max_user = -1

        for i in range(len(adjacency_test)):
            user_total = np.sum(adjacency_test[i])
            if user_total > max_total:
                max_total = user_total
                max_user = i

        num_classes = max(y_train.flatten()) + 1
        max_size = max_size_matrices
        learning_rate = 0.001

        user_index = max_user
        self.heatmap_matrices(str(fold_number), [adjacency_test[user_index], adjacency_test_week[user_index],
                                                 adjacency_test_weekend[user_index],
                                                 temporal_test[user_index], temporal_test_week[user_index],
                                                 temporal_test_weekend[user_index], location_time_test[user_index],
                                                 location_location_test[user_index]
                                                 ],
                              [
                                  "Adjacency",
                                  "Adjacency (weekday)",
                                  "Adjacency (weekend)",
                                  "Temporal",
                                  "Temporal (weekday)", "Temporal (weekend)", "Location_time", "Location_location"],
                              output_dir)


        input_train = [adjacency_train, adjacency_week_train, adjacency_train_weekend,
                       temporal_train, temporal_train_week, temporal_train_weekend, distance_train,
                       duration_train, location_time_train, location_location_train]

        input_test = [adjacency_test, adjacency_test_week, adjacency_test_weekend, temporal_test, temporal_test_week,
                      temporal_test_weekend,
                      distance_test, duration_test, location_time_test, location_location_test]


        np.save("temporal_train.npy", temporal_train)
        np.save("a_input.npy", adjacency_train)


        # verifying whether categories arrays are equal
        compare1 = y_train == y_train_week
        compare2 = y_train_week == y_train_weekend
        compare3 = y_test == y_test_week
        compare4 = y_test_week == y_test_weekend

        if not (compare1.all() and compare2.all() and compare3.all() and compare4.all()):
            print("\nListas difernetes de categorias\n")
            exit()

        # Model setup and hyperparameters
        self.device = torch.device(Configurations.DEVICE)
        self.model = GNNUS_BaseModel(num_classes, max_size, max_size_sequence, self.features_num_columns).to(
            self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = torch.nn.CrossEntropyLoss()  # Automatically applies Softmax for you

        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=100, verbose=True)  #
        self.batch_size = max_size * 4
        print("\nTamanho do batch_size: ", self.batch_size)

        # TOSHOW: Será que usar o LabelEncoder ao inves de np_utils.to_categorical
        # Converting y_train and y_test to tensor and one-hot encoding
        y_train_encoded = np_utils.to_categorical(y_train, num_classes=num_classes)
        y_test_encoded = np_utils.to_categorical(y_test, num_classes=num_classes)

        y_train_tensor = torch.tensor(y_train_encoded, dtype=torch.float32).to(self.device)
        y_test_tensor = torch.tensor(y_test_encoded, dtype=torch.float32).to(self.device)

        # Creating data loaders
        input_train_tensor = [torch.tensor(data, dtype=torch.float32).to(self.device) for data in input_train]
        input_test_tensor = [torch.tensor(data, dtype=torch.float32).to(self.device) for data in input_test]

        train_dataset = TensorDataset(*input_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(*input_test_tensor, y_test_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        # Training the model
        h = self.train_model(train_loader, test_loader, epochs)
        report, accuracy = self.evaluate_model(test_loader)

        # Append the history metrics
        h['acc'].append(accuracy)
        h['val_acc'].append(accuracy)

        return h, report, model, accuracy

    # Training loop
    def train_model(self, train_loader, test_loader, epochs):
        # History storage
        history = {
            'acc': [],
            'loss': [],
            'val_acc': [],
            'val_loss': []
        }

        self.model.train()

        for epoch in range(epochs):
            # Reset history metrics
            running_loss = 0.0
            correct = 0
            total = 0
            y_pred = []
            y_true = []
            for loader in train_loader:
                self.optimizer.zero_grad()

                inputs, targets = loader[:-1], loader[-1].float()
                outputs = self.model(*inputs)
                outputs = outputs.view(targets.size(0), targets.size(1), targets.size(2))

                loss = self.loss_fn(outputs, targets)
                loss.backward()
                self.optimizer.step()

                # Store the loss
                running_loss += loss.item()
                total += targets.size(0)

                y_pred.extend(one_hot_decoding_predicted(outputs.cpu().detach().numpy()))
                y_true.extend(one_hot_decoding_predicted(targets.cpu().detach().numpy()))

                report = classification_report(y_true, y_pred, output_dict=True)
                correct = report['accuracy']

            # TOSHOW
            # Update learning rate based on the training loss
            self.scheduler.step(loss)

            # Store training metrics
            epoch_loss = running_loss / len(train_loader)
            epoch_acc = correct
            history['loss'].append(epoch_loss)
            history['acc'].append(epoch_acc)

            # Validation step
            self.model.eval()

            # Reset metrics
            val_running_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():

                for loaders in test_loader:
                    inputs, targets = loaders[:-1], loaders[-1].float()

                    inputs, targets = [i.to(self.device) for i in inputs], targets.to(self.device)

                    outputs = self.model(*inputs)
                    outputs = outputs.view(targets.size(0), targets.size(1), targets.size(2))

                    loss = self.loss_fn(outputs, targets)
                    val_running_loss += loss.item()
                    val_total += targets.size(0)



                    #y_true.extend(targets.argmax(dim=-1).cpu().numpy().float().mean().item())

                    #y_pred.extend(one_hot_decoding_predicted(np.array(inputs)))
                    #y_true.extend(one_hot_decoding_predicted(np.array(targets)))


                    val_correct += (outputs.argmax(dim=-1) == targets.argmax(dim=-1)).float().mean().item()

            val_epoch_loss = val_running_loss / len(test_loader)
            val_epoch_acc =  100* val_correct/val_total
            history['val_loss'].append(val_epoch_loss)
            history['val_acc'].append(val_epoch_acc)

            print(
                f'Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}, Val_Loss: {val_epoch_loss:.4f}, Val_Acc: {val_epoch_acc:.2f}%')

        return history

    def evaluate_model(self, test_loader):
        self.model.eval()

        with torch.no_grad():
            y_pred = []
            y_true = []
            for loaders in test_loader:
                inputs, targets = loaders[:-1], loaders[-1].float()

                outputs = self.model(*inputs)
                outputs = outputs.view(targets.size(0), targets.size(1), targets.size(2))

                y_pred.extend(one_hot_decoding_predicted(outputs.cpu().numpy()))
                y_true.extend(one_hot_decoding_predicted(targets.cpu().numpy()))


        report = classification_report(y_true, y_pred, output_dict=True)
        print(report)
        return report, report['accuracy']

    def heatmap_matrices(self, fold_number, matrices, names, output_dir):

        for matrix, name in zip(matrices, names):
            self.poi_categorization_loader.heatmap(output_dir, matrix, name.replace(" ", "_") + "_" + fold_number, name,
                                                   (10, 10), True)

    def _add_location_report(self, location_report, report):
        for l_key in report:
            if l_key == 'accuracy':
                location_report[l_key].append(report[l_key])
                continue
            for v_key in report[l_key]:
                location_report[l_key][v_key].append(report[l_key][v_key])

        return location_report

    def _find_best_model(self, models, accuracies):

        index = np.argmax(accuracies)
        return models[index]

    def preprocess_report(self, report, int_to_categories):

        new_report = {}

        for key in report:
            if key != 'accuracy' and key != 'macro avg' and key != 'weighted avg':
                new_report[int_to_categories[key]] = report[key]
            else:
                new_report[key] = report[key]

        return new_report

    def _min_max_normalize(self, matrix):

        matrix_1 = matrix.transpose()
        scaler = MinMaxScaler()
        scaler.fit(matrix_1)
        matrix_1 = scaler.transform(matrix_1).transpose()

        return matrix_1