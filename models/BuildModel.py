from models.CNN1D_1 import CNN1D
from models.BiLSTM import BiLSTM
from models.BiLSTMCNN1d import BiLSTMCNN1d
from models.CNN1d import CNN1d
from models.CNN1dTransformer import CNN1dTransformer
from models.CNNTransformer import CNNADFormer
from models.CNNTransformer2 import CNNADFormer2
from models.CNNTransformer3 import CNNADFormer3
from models.CNNTransformerCNN1d import CNNADFormerCNN1d
from models.Deep_packet import DeepPacket
from models.LSTM import LSTM
from models.LSTMCNN1d import LSTMCNN1d
from models.Linear_Transformer import LinearADFormer
from models.RNN import RNN
from models.RNNCNN1d import RNNCNN1d
from models.SVM import SVM
from models.Transformer import Transformer
from models.TransformerCNN1d import TransformerCNN1d
from models.TransformerCNN2D import TransformerCNN2D
from models.WaveTransformer1 import WaveletADFormer1
from models.WaveTransformer2 import WaveletADFormer2
from models.WaveTransformer3 import WaveletADFormer3
from models.WaveTransformerCNN1D import WaveTransformerCNN1D
from models.swin_transformer_v2 import SwinTransformerV2


def build(model_name, attention_type,feature_num,e_layers,num_heads,num_classes,win_size,device,dropout,d_model,kernel_size,padding,wavelet_method,trainable):
    if model_name == 'Linear_Transformer':
        return LinearADFormer(
            attention_type=attention_type,
            feature_num=feature_num,
            e_layers = e_layers,
            num_heads = num_heads,
            num_classes = num_classes,
            device= device,
            dropout = dropout,
            d_model =  d_model,
        )
    elif model_name == 'TransformerCNN1d':
        return TransformerCNN1d(
            attention_type=attention_type,
            feature_num=feature_num,
            e_layers = e_layers,
            num_heads = num_heads,
            num_classes = num_classes,
            device= device,
            win_size=win_size,
            dropout = dropout,
            d_model =  d_model,
        )
    elif model_name == 'CNN1dTransformer':
        return CNN1dTransformer(
            attention_type=attention_type,
            feature_num=feature_num,
            e_layers = e_layers,
            num_heads = num_heads,
            num_classes = num_classes,
            device= device,
            win_size=win_size,
            dropout = dropout,
            d_model =  d_model,
        )
    elif model_name == 'CNN_Transformer':
        return CNNADFormer(
            attention_type=attention_type,
            feature_num=feature_num,
            e_layers = e_layers,
            num_heads = num_heads,
            num_classes = num_classes,
            kernel_size= kernel_size,
            padding = padding,
            device= device,
            dropout = dropout,
            d_model =  d_model,
        )
    elif model_name == 'CNN_Transformer2':
        return CNNADFormer2(
            attention_type=attention_type,
            feature_num=feature_num,
            e_layers = e_layers,
            num_heads = num_heads,
            num_classes = num_classes,
            win_size=win_size,
            kernel_size= kernel_size,
            padding = padding,
            device= device,
            dropout = dropout,
            d_model =  d_model,
        )
    elif model_name == 'CNN_Transformer3':
        return CNNADFormer3(
            attention_type=attention_type,
            feature_num=feature_num,
            e_layers = e_layers,
            num_heads = num_heads,
            num_classes = num_classes,
            win_size=win_size,
            kernel_size= kernel_size,
            padding = padding,
            device= device,
            dropout = dropout,
            d_model =  d_model,
        )
    elif model_name == 'CNN_Transformer_CNN1d':
        return CNNADFormerCNN1d(
            attention_type=attention_type,
            feature_num=feature_num,
            e_layers = e_layers,
            num_heads = num_heads,
            num_classes = num_classes,
            kernel_size= kernel_size,
            padding = padding,
            device= device,
            win_size = win_size,
            dropout = dropout,
            d_model =  d_model,
        )
    elif model_name == 'Wavelet_Transformer1':
        return WaveletADFormer1(
            attention_type=attention_type,
            feature_num=feature_num,
            e_layers = e_layers,
            num_heads = num_heads,
            num_classes = num_classes,
            win_size = win_size,
            device= device,
            dropout = dropout,
            d_model =  d_model,
            wavelet_method = wavelet_method,
            trainable = trainable
        )
    elif model_name == 'Wavelet_Transformer2':
        return WaveletADFormer2(
            attention_type=attention_type,
            feature_num=feature_num,
            e_layers = e_layers,
            num_heads = num_heads,
            num_classes = num_classes,
            win_size = win_size,
            device= device,
            dropout = dropout,
            d_model =  d_model,
            wavelet_method = wavelet_method,
            trainable = trainable
        )
    elif model_name == 'Wavelet_Transformer3':
        return WaveletADFormer3(
            attention_type=attention_type,
            feature_num=feature_num,
            e_layers = e_layers,
            num_heads = num_heads,
            num_classes = num_classes,
            win_size = win_size,
            device= device,
            dropout = dropout,
            d_model =  d_model,
            wavelet_method = wavelet_method,
            trainable = trainable
        )
    elif model_name == 'BiLSTMCNN1d':
        return BiLSTMCNN1d(
            feature_num=feature_num,
            e_layers = e_layers,
            d_model = d_model,
            num_classes = num_classes,
            device= device,
            win_size = win_size,
            dropout = dropout
        )
    elif model_name == 'CNN1d':
        return CNN1d(
            feature_num=feature_num,
            e_layers = e_layers,
            d_model = d_model,
            num_classes = num_classes,
            win_size=win_size,
            device= device,
            dropout = dropout
        )
    elif model_name == 'BiLSTM':
        return BiLSTM(
            feature_num=feature_num,
            e_layers = e_layers,
            d_model = d_model,
            num_classes = num_classes,
            device= device,
            win_size = win_size,
            dropout = dropout
        )
    elif model_name == 'LSTM':
        return LSTM(
            feature_num=feature_num,
            e_layers = e_layers,
            d_model = d_model,
            num_classes = num_classes,
            win_size = win_size,
            device= device,
            dropout = dropout
        )
    elif model_name == 'LSTMCNN1d':
        return LSTMCNN1d(
            feature_num=feature_num,
            e_layers = e_layers,
            d_model = d_model,
            num_classes = num_classes,
            device= device,
            win_size = win_size,
            dropout = dropout
        )
    elif model_name == 'RNN':
        return RNN(
            feature_num=feature_num,
            e_layers = e_layers,
            d_model = d_model,
            num_classes = num_classes,
            win_size = win_size,
            device= device,
            dropout = dropout
        )
    elif model_name == 'RNNCNN1d':
        return RNNCNN1d(
            feature_num=feature_num,
            e_layers = e_layers,
            d_model = d_model,
            num_classes = num_classes,
            device= device,
            win_size = win_size,
            dropout = dropout
        )
    elif model_name == 'SVM':
        return SVM(
            feature_num=feature_num,
            e_layers=e_layers,
            d_model=d_model,
            num_classes = num_classes,
            device=device,
            dropout=dropout
        )
    elif model_name == 'Transformer':
        return Transformer(
            feature_num=feature_num,
            e_layers=e_layers,
            num_heads=num_heads,
            num_classes=num_classes,
            win_size=win_size,
            device=device,
            dropout=dropout,
            d_model=d_model,
        )
    elif model_name == 'WaveTransformerCNN1D':
        return WaveTransformerCNN1D(
            attention_type=attention_type,
            feature_num=feature_num,
            e_layers=e_layers,
            num_heads=num_heads,
            num_classes=num_classes,
            win_size=win_size,
            device=device,
            dropout=dropout,
            d_model=d_model,
            wavelet_method=wavelet_method,
            trainable=trainable
        )
    elif model_name == 'TransformerCNN2D':
        return TransformerCNN2D(
            attention_type=attention_type,
            feature_num=feature_num,
            e_layers = e_layers,
            num_heads = num_heads,
            num_classes = num_classes,
            device= device,
            win_size=win_size,
            dropout = dropout,
            d_model =  d_model,
        )
    elif model_name == 'CNN1D':
        return CNN1D(
            num_classes = num_classes
        )
    elif model_name == 'DeepPacket':
        return DeepPacket(
            num_classes = num_classes
        )
    elif model_name == 'SWinFormer':
        return SwinTransformerV2()
    else:
        print("No such model!")
        return None
