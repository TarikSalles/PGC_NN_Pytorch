import torch
import torch.nn as nn
import torch.nn.functional as F
from  torch_geometric.nn.conv import ARMAConv



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
    def __init__(self, classes, max_size_matrices, max_size_sequence, features_num_columns: int):
        super(GNNUS_BaseModel, self).__init__()
        self.max_size_matrices = max_size_matrices
        self.max_size_sequence = max_size_sequence
        self.classes = classes
        self.features_num_columns = features_num_columns

        self.out_temporal = ARMAConv(in_channels=-1, out_channels=20,shared_weights=share_weights,  act=nn.GELU(), dropout=dropout_skip)


        self.out_week_temporal = ARMAConv(in_channels=-1,out_channels =20, act=nn.GELU(), shared_weights=share_weights,
                                     dropout=dropout_skip)

        self.out_weekend_temporal = ARMAConv(in_channels=-1,out_channels =20, act=nn.GELU(), shared_weights=share_weights,
                                     dropout=dropout_skip)

        self.out_distance = ARMAConv(in_channels=-1,out_channels =20, act=nn.GELU(), shared_weights=share_weights,
                                     dropout=dropout_skip)

        self.out_duration = ARMAConv(in_channels=-1,out_channels =20, act=nn.GELU(), shared_weights=share_weights,
                                     dropout=dropout_skip)

        self.out_location_location = ARMAConv(in_channels=-1,out_channels =20, act=nn.GELU(), shared_weights=share_weights,
                                     dropout=dropout_skip)

        self.out_location_time = nn.Linear(48, 40) #Pytorch não possui camadas Dense portanto devem ser lineares
        self.out_dense = nn.Linear(self.classes, self.classes) #Pytorch não possui camadas Dense portanto devem ser lineares
        self.out_gnn = nn.Linear(self.classes, self.classes) #Pytorch não possui camadas Dense portanto devem ser lineares

    def forward(self, A_input, A_week_input, A_weekend_input, Temporal_input, Temporal_week_input,
                Temporal_weekend_input, Distance_input, Duration_input, Location_time_input, Location_location_input):

        A_input = A_input.view(2, -1) #Matrizes de adjacência devem possuir formato (2,E) na ARMAConv
        A_input = A_input.to(torch.int64) #Matrizes de adjacência devem ser somente inteiras

        Temporal_input = Temporal_input.to(torch.float32)
        Temporal_input = Temporal_input.permute(1, 0, 2) #Permutação da matriz pois a ARMAConv requer que o numero de vertices venha primeiro

        elu_activation = nn.ELU() #Objeto de ativação ELU
        softmax_activation = nn.Softmax() #Objeto de ativação Softmax

        out_temporal = elu_activation(self.out_temporal(Temporal_input,A_input)) #Ativação ELU chamada
        out_temporal = F.dropout(out_temporal, p=0.3) #Dropout
        arma_out_temporal = ARMAConv(in_channels=-1, out_channels=self.classes,shared_weights=share_weights, dropout=dropout_skip) #Segunda camada ARMAConv sem ativação distinta
        out_temporal = out_temporal.to(torch.float32)
        out_temporal = softmax_activation(arma_out_temporal(out_temporal,A_input)) #Ativação Sofmax chamada

        A_week_input = A_week_input.view(2, -1) #Matrizes de adjacência devem possuir formato (2,E) na ARMAConv
        A_week_input = A_week_input.to(torch.int64) #Matrizes de adjacência devem ser somente inteiras
        Temporal_week_input = Temporal_week_input.to(torch.float32)
        Temporal_week_input = Temporal_week_input.permute(1, 0, 2) #Permutação da matriz pois a ARMAConv requer que o numero de vertices venha primeiro

        out_week_temporal = elu_activation(self.out_week_temporal(Temporal_week_input, A_week_input)) #Ativação ELU chamada
        out_week_temporal = F.dropout(out_week_temporal, p=0.3) #Dropout
        arma_out_week_temporal = ARMAConv(in_channels=-1,out_channels=self.classes,shared_weights=share_weights, dropout=dropout_skip) #Segunda camada ARMAConv sem ativação distinta
        out_week_temporal = out_week_temporal.to(torch.float32)
        out_week_temporal = softmax_activation(arma_out_week_temporal(out_week_temporal,A_week_input)) #Ativação Sofmax chamada

        A_weekend_input = A_weekend_input.view(2, -1) #Matrizes de adjacência devem possuir formato (2,E) na ARMAConv
        A_weekend_input = A_weekend_input.to(torch.int64) #Matrizes de adjacência devem ser somente inteiras
        Temporal_weekend_input = Temporal_weekend_input.to(torch.float32)
        Temporal_weekend_input = Temporal_weekend_input.permute(1, 0, 2) #Permutação da matriz pois a ARMAConv requer que o numero de vertices venha primeiro


        out_weekend_temporal = elu_activation(self.out_weekend_temporal(Temporal_weekend_input, A_weekend_input)) #Ativação ELU chamada
        out_weekend_temporal = F.dropout(out_weekend_temporal, p=0.3) #Dropout
        arma_out_weekend_temporal = ARMAConv(in_channels=-1,out_channels=self.classes,shared_weights=share_weights, dropout=dropout_skip)  #Segunda camada ARMAConv sem ativação distinta
        out_weekend_temporal = out_weekend_temporal.to(torch.float32)
        out_weekend_temporal = softmax_activation(arma_out_weekend_temporal(out_weekend_temporal,A_weekend_input)) #Ativação Softmax chamada

        Distance_input = Distance_input.to(torch.float32)
        Distance_input = Distance_input.permute(1, 0, 2) #Permutação da matriz pois a ARMAConv requer que o numero de vertices venha primeiro


        out_distance = elu_activation(self.out_distance(Distance_input, A_input)) #Ativação ELU chamada
        out_distance = F.dropout(out_distance, p=0.3) #Dropout
        arma_out_distance = ARMAConv(in_channels=-1,out_channels=self.classes,shared_weights=share_weights, dropout=dropout_skip)  #Segunda camada ARMAConv sem ativação distinta
        out_distance = out_distance.to(torch.float32)
        out_distance = softmax_activation(arma_out_distance(out_distance, A_input)) #Ativação Softmax chamada

        Duration_input = Duration_input.to(torch.float32)
        Duration_input = Duration_input.permute(1, 0, 2) #Permutação da matriz pois a ARMAConv requer que o numero de vertices venha primeiro


        out_duration = elu_activation(self.out_duration(Duration_input, A_input)) #Ativação ELU chamada
        out_duration = F.dropout(out_duration, p=0.3) #Dropout
        arma_out_duration = ARMAConv(in_channels=-1,out_channels=self.classes,shared_weights=share_weights, dropout=dropout_skip) #Segunda camada ARMAConv sem ativação distinta

        out_duration = out_duration.to(torch.float32)
        out_duration = softmax_activation(arma_out_duration(out_duration,A_input)) #Ativação Softmax chamada

        Location_location_input = Location_location_input.view(2, -1) #Matrizes de adjacência devem possuir formato (2,E) na ARMAConv
        Location_location_input = Location_location_input.to(torch.int64) #Matrizes de adjacência devem ser somente inteiras
        Location_time_input = Location_time_input.to(torch.float32)
        Location_time_input = Location_time_input.permute(1, 0, 2) #Permutação da matriz pois a ARMAConv requer que o numero de vertices venha primeiro


        out_location_location = elu_activation(self.out_location_location(Location_time_input, Location_location_input)) #Ativação ELU chamada
        out_location_location = F.dropout(out_location_location, p=0.3) #Dropout
        arma_out_location_location = ARMAConv(in_channels=-1,out_channels=self.classes,shared_weights=share_weights, dropout=dropout_skip) #Segunda camada ARMAConv sem ativação distinta

        out_location_location = out_location_location.to(torch.float32)
        out_location_location = softmax_activation(arma_out_location_location(out_location_location,Location_location_input))  #Ativação Softmax chamada

        out_location_time = self.out_location_time(Location_time_input) #Camada linear
        out_location_time = F.dropout(out_location_time, p=0.3) #Dropout
        Linear_layer = nn.Linear(40, self.classes) #Camada linear

        out_location_time = softmax_activation(Linear_layer(out_location_time))  #Ativação Softmax chamada


        out_dense = 2.0 * out_location_location + 2.0 * out_location_time #Multiplicação com pesos
        out_dense = softmax_activation(self.out_dense(out_dense))  #Ativação Softmax chamada

        out_gnn = 1.0 * out_temporal + 1.0 * out_week_temporal + 1.0 * out_weekend_temporal + 1.0 * out_distance + 1.0 * out_duration #Multiplicação com pesos
        out_gnn = softmax_activation(self.out_gnn(out_gnn))  #Ativação Softmax chamada


        out = 1.0 * out_dense + 1.0 * out_gnn #Multiplicação com pesos
        out = out.permute(1, 2, 0)  ##Permutação da matriz tridimensional out afim de seguir o formato requisitada

        return  out #Retorno da camada final
