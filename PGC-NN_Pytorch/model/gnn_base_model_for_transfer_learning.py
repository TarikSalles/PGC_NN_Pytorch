import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import ARMAConv

from utils.nn_preprocessing import prepare_pyg_batch
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.data import Data
from  torch_geometric.utils import get_laplacian
iterations = 1  # Number of iterations to approximate each ARMA(1)
order = 1  # Order of the ARMA filter (number of parallel stacks)
share_weights = True  # Share weights in each ARMA stack
dropout = 0.5  # Dropout rate applied between layers
dropout_skip = 0.3  # Dropout rate for the internal skip connection of ARMA
l2_reg = 5e-5  # L2 regularization rate
learning_rate = 1e-2  # Learning rate
epochs = 15  # Number of training epochs
es_patience = 100  # Patience for early stopping


class GNNUS_BaseModel(nn.Module):
    def __init__(self, batch_size,classes, max_size_matrices, max_size_sequence, features_num_columns: int):
        super(GNNUS_BaseModel, self).__init__()

        self.max_size_matrices = max_size_matrices
        self.max_size_sequence = max_size_sequence
        self.classes = classes
        self.features_num_columns = features_num_columns

        # Week_total
        self.arma_conv_temporal = ARMAConv(-1, 20,
                                           num_stacks=1,  # ?
                                           num_layers=1,  # Order
                                           act=nn.GELU(),
                                           shared_weights=share_weights,
                                           dropout=dropout_skip)
        self.dropout_temporal = nn.Dropout(0.3)  # Dropout
        self.arma_conv_final_temporal = ARMAConv(-1, self.classes)

        # Week_temporal
        self.arma_conv_temporal_week = ARMAConv(-1, 20,
                                                num_stacks=1,  # ?
                                                num_layers=1,  # Order
                                                act=nn.GELU(),
                                                shared_weights=share_weights,
                                                dropout=dropout_skip)
        self.dropout_temporal_week = nn.Dropout(0.3)
        self.arma_conv_final_temporal_week = ARMAConv(-1, self.classes)

        # Weekend_temporal
        self.arma_conv_temporal_weekend = ARMAConv(-1, 20,
                                                   num_stacks=1,  # ?
                                                   num_layers=1,  # Order
                                                   act=nn.GELU(),
                                                   shared_weights=share_weights,
                                                   dropout=dropout_skip)
        self.dropout_temporal_weekend = nn.Dropout(0.3)
        self.arma_conv_final_temporal_weekend = ARMAConv(-1, self.classes)

        # Distance
        self.arma_conv_distance = ARMAConv(-1, 20,
                                           num_stacks=1,  # ?
                                           num_layers=1,  # Order
                                           act=nn.GELU())
        self.dropout_distance = nn.Dropout(0.3)
        self.arma_conv_final_distance = ARMAConv(-1, self.classes)

        # Duration
        self.arma_conv_duration = ARMAConv(-1, 20,
                                           num_stacks=1,  # ?
                                           num_layers=1,  # Order
                                           act=nn.GELU())
        self.dropout_duration = nn.Dropout(0.3)
        self.arma_conv_final_duration = ARMAConv(-1, self.classes)

        # Location_time
        self.arma_conv_location_time = ARMAConv(-1, 20,
                                                num_stacks=1,  # ?
                                                num_layers=1,  # Order
                                                act=nn.GELU())
        self.dropout_location_time = nn.Dropout(0.3)
        self.arma_conv_final_location_time = ARMAConv(-1, self.classes)

        self.dense_location_time = nn.Sequential(
            nn.Linear(self.features_num_columns, 40),
            nn.ReLU(),
            nn.Linear(40, classes),
            nn.Softmax(),
        )

        # Output layers
        self.dense_location_location = nn.Sequential(
            nn.Linear(classes, classes),
          #  nn.Softmax(),
        )

        self.output_gnn = nn.Sequential(
            nn.Linear(classes, classes),
           # nn.Softmax(),
        )

    def forward(self, A_input, A_week_input, A_weekend_input, Temporal_input, Temporal_week_input,
                Temporal_weekend_input, Distance_input, Duration_input, Location_time_input, Location_location_input):
        transform = NormalizeFeatures()
        
        
        
        A_input, A_input_weights = prepare_pyg_batch(A_input)
        A_week_input, A_week_input_weights = prepare_pyg_batch(A_week_input)
        A_weekend_input, A_weekend_input_weights = prepare_pyg_batch(A_weekend_input)
        Location_location_input, Location_location_input_weights = prepare_pyg_batch(Location_location_input)

       # A_input, A_input_weights = get_laplacian(A_input,A_input_weights, normalization='sym')
        #A_week_input, A_week_input_weights = get_laplacian(A_week_input, A_week_input_weights, normalization='sym')
        #A_weekend_input, A_weekend_input_weights = get_laplacian(A_weekend_input, A_weekend_input_weights, normalization='sym')
        #Location_location_input, Location_location_input_weights = get_laplacian(Location_location_input, Location_location_input_weights, normalization='sym')

        '''''
        A_input_Data = Data(x=A_input.to(torch.float), edge_index=A_input_weights)
        A_week_Data = Data(x=A_week_input.to(torch.float), edge_index=A_week_input_weights)
        A_weekend_Data = Data(x=A_weekend_input.to(torch.float), edge_index=A_weekend_input_weights)
        Location_location_Data =  Data(x=Location_location_input.to(torch.float), edge_index=Location_location_input_weights)


        A_input_transform = transform(A_input_Data)
        A_week_transform = transform(A_week_Data)
        A_weekend_transform = transform(A_weekend_Data)
        Location_Location_transform = transform(Location_location_Data)
        '''''

        
        #A_input, A_input_weights = A_input_transform.x.to(torch.int64), A_input_transform.edge_index
        #A_week_input, A_week_input_weights = A_week_transform.x.to(torch.int64), A_week_transform.edge_index
        #A_weekend_input, A_weekend_input_weights = A_weekend_transform.x.to(torch.int64), A_weekend_transform.edge_index
        #Location_location_input, Location_location_input_weights = Location_Location_transform.x.to(torch.int64), Location_Location_transform.edge_index

        Temporal_input = Temporal_input.view(Temporal_input.size(0) * Temporal_input.size(1), Temporal_input.size(2))
        Temporal_week_input = Temporal_week_input.view(Temporal_week_input.size(0) * Temporal_week_input.size(1),
                                                       Temporal_week_input.size(2))
        Temporal_weekend_input = Temporal_weekend_input.view(
            Temporal_weekend_input.size(0) * Temporal_weekend_input.size(1), Temporal_weekend_input.size(2))
        Distance_input = Distance_input.view(Distance_input.size(0) * Distance_input.size(1), Distance_input.size(2))
        Duration_input = Duration_input.view(Duration_input.size(0) * Duration_input.size(1), Duration_input.size(2))
        Location_time_input = Location_time_input.view(Location_time_input.size(0) * Location_time_input.size(1),
                                                       Location_time_input.size(2))
        
        out_temporal = F.elu(self.arma_conv_temporal(Temporal_input, A_input, A_input_weights))
        out_temporal = self.dropout_temporal(out_temporal)
        out_temporal = F.softmax(self.arma_conv_final_temporal(out_temporal, A_input, A_input_weights))

        out_week_temporal = F.elu(self.arma_conv_temporal_week(Temporal_week_input, A_week_input, A_week_input_weights))
        out_week_temporal = self.dropout_temporal_week(out_week_temporal)
        out_week_temporal = F.softmax(
            self.arma_conv_final_temporal_week(out_week_temporal, A_week_input, A_week_input_weights))

        out_weekend_temporal = F.elu(
            self.arma_conv_temporal_weekend(Temporal_weekend_input, A_weekend_input, A_weekend_input_weights))
        out_weekend_temporal = self.dropout_temporal_weekend(out_weekend_temporal)
        out_weekend_temporal = F.softmax(
            self.arma_conv_final_temporal_weekend(out_weekend_temporal, A_weekend_input, A_weekend_input_weights))

        out_distance = F.elu(self.arma_conv_distance(Distance_input, A_input, A_input_weights))
        out_distance = self.dropout_distance(out_distance)
        out_distance = F.softmax(self.arma_conv_final_distance(out_distance, A_input, A_input_weights))

        out_duration = F.elu(self.arma_conv_duration(Duration_input, A_input, A_input_weights))
        out_duration = self.dropout_duration(out_duration)
        out_duration = F.softmax(self.arma_conv_final_duration(out_duration, A_input, A_input_weights))

        
        out_location_location = F.elu(
            self.arma_conv_location_time(Location_time_input, Location_location_input, Location_location_input_weights))
        out_location_location = self.dropout_location_time(out_location_location)
        out_location_location = F.softmax(self.arma_conv_final_location_time(out_location_location, Location_location_input,
                                                                         Location_location_input_weights))

        out_location_time = self.dense_location_time(Location_time_input)

        out_dense = (torch.tensor(2.) * out_location_location) + (torch.tensor(2.) * out_location_time)

        out_dense = self.dense_location_location(out_dense)

        out_gnn = (
                (torch.tensor(1.) * out_temporal)
                + (torch.tensor(1.) * out_week_temporal)
                + (torch.tensor(1.) * out_weekend_temporal)
                + (torch.tensor(1.) * out_distance)
                + (torch.tensor(1.) * out_duration)
        )
        out_gnn = self.output_gnn(out_gnn)
        out = (
                (torch.tensor(1.) * out_dense)
                + (torch.tensor(1.) * out_gnn)
        )
        return out
