from foundations import hparams
from losses import loss_functions
from lottery.desc import LotteryDesc
from models import base
from pruning import sparse_global
import torch
import torch.nn as nn
import torch.nn.functional as F

_3D = '3d'
_2D = '2d'
_CLASSIFICATION = 'classification'
_SEGMENTATION = 'segmentation'
_HIGH_RES = 'high_res'
_LOW_RES = 'low_res'
_DEFAULT_RES = 'default_res'

class Model(base.Model):
    '''A NCA model for segmentation of 2D and 3D data'''
    '''Use it like this: nca_<2d/3d>_<segmentation/classification>'''

    def __init__(self, plan, initializer, model_hparams: hparams.ModelHparams, outputs=2, res=_DEFAULT_RES):
        super(Model, self).__init__()
        self.state_channel_num = model_hparams.state_channel_num
        self.fire_rate = model_hparams.fire_rate
        self.hidden_size = model_hparams.hidden_size
        self.input_channel_num = model_hparams.input_channel_num
        self.steps = model_hparams.nca_steps if res == _DEFAULT_RES else (model_hparams.nca_steps_high_res if res == _HIGH_RES else model_hparams.nca_steps_low_res)
        self.dimensions = plan[0]
        self.task = plan[1]

        convolution = nn.Conv2d if self.dimensions == _2D else nn.Conv3d
        batchnorm = nn.BatchNorm2d if self.dimensions == _2D else nn.BatchNorm3d

        # Perception
        self.perception_conv0 = convolution(self.state_channel_num, self.state_channel_num, kernel_size=3, stride=1, padding=1, groups=self.state_channel_num, padding_mode="reflect")
        self.perception_conv1 = convolution(self.state_channel_num, self.state_channel_num, kernel_size=3, stride=1, padding=1, groups=self.state_channel_num, padding_mode="reflect")

        # Update
        self.fc0 = convolution(self.state_channel_num*3, self.hidden_size, kernel_size=1) #*3 because we have 2 convolutions and 1 identity
        self.bn = batchnorm(self.hidden_size, track_running_stats=False) #running stats false to avoid same values for each nca step
        self.fc1 = convolution(self.hidden_size, self.state_channel_num, kernel_size=1)

        # Classifier
        if self.task == _CLASSIFICATION:
            self.classifier_fc0 = convolution(self.state_channel_num, self.hidden_size, kernel_size=1)
            self.classifier_fc1 = convolution(self.hidden_size, outputs, kernel_size=1)

        #TODO why should init fc1 with zeros?

        if self.task == _SEGMENTATION:
            self.criterion = loss_functions.DiceCELoss()
        else:
            self.criterion = torch.nn.CrossEntropyLoss()

        self.apply(initializer) #TODO how does initializer work?
	
    def perceive(self, x):
        CHANNEL_DIM=1
        z1 = self.perception_conv0(x)
        z2 = self.perception_conv1(x)
        y = torch.cat((x,z1,z2),CHANNEL_DIM)
        return y
    
    def update(self, x):
        dx = self.perceive(x)
        dx = self.fc0(dx)
        dx = self.bn(dx) # batch norm against gradient vanishing / explosion
        dx = F.relu(dx)
        dx = self.fc1(dx)

        if self.fire_rate < 1.0:
            with torch.no_grad():
                stochastic_shape = [dx.size(0),1,dx.size(2),dx.size(3)] if self.dimensions == _2D else [dx.size(0),1,dx.size(2),dx.size(3), dx.size(4)]
                stochastic = torch.rand(stochastic_shape, device=dx.device)<self.fire_rate
            dx = dx * stochastic

        x = x + dx
        return x
    
    def forward(self, x):
        # Image padding to increase the number of channels
        CHANNEL_DIM = 1
        if x.shape[CHANNEL_DIM] < self.state_channel_num:
            pad_channels = self.state_channel_num - x.shape[CHANNEL_DIM]
            zeros = torch.zeros(x.shape[0], pad_channels, x.shape[2], x.shape[3], device=x.device) if self.dimensions == _2D else torch.zeros(x.shape[0], pad_channels, x.shape[2], x.shape[3], x.shape[4], device=x.device)
            x = torch.cat([x, zeros], dim=CHANNEL_DIM)
        
        for step in range(self.steps):
            x_updated = self.update(x)
            # keep input image, only update hidden state channels
            x = torch.concat((x[:, :self.input_channel_num, ...], x_updated[:, self.input_channel_num:, ...]), CHANNEL_DIM)

        if self.task == _CLASSIFICATION:
            print("-- before mean:" + str(x.shape))
            if self.dimensions == _3D:
                x = x.mean([2, 3, 4]) # TODO use max pooling?
            else:
                x = x.mean([2, 3]) # TODO use max pooling?
            print("-- after mean:" + str(x.shape))
            x = self.classifier_fc0(x)
            x = F.relu(x)
            x = self.classifier_fc1(x)

        return x

    @property
    def output_layer_names(self):
        #TODO what is this
        if self.task == _CLASSIFICATION:
            return ['classifier_fc1.weight', 'classifier_fc1.bias']
        return ['fc1.weight', 'fc1.bias'] # TODO this can't be right as these layers are in the update function

    @staticmethod
    def is_valid_model_name(model_name):
        parts = model_name.split('_')
        if len(parts) >= 3:
            return parts[0] == 'nca' and parts[1] in [_2D, _3D] and parts[2] in [_SEGMENTATION, _CLASSIFICATION]
        return False

    @staticmethod
    def get_model_from_name(model_name, initializer, outputs=None, model_hparams=None):
        parts = model_name.split('_')
        plan = [parts[1], parts[2]]
        return Model(plan, initializer, model_hparams=model_hparams, outputs=outputs)

    @property
    def loss_criterion(self):
        return self.criterion

    @staticmethod
    def default_hparams():
        model_hparams = hparams.ModelHparams(
            model_name='nca',
            model_init='kaiming_normal',
            batchnorm_init='uniform',
            nca_steps=32,
            hidden_size=64,
            fire_rate=0.5,
            input_channel_num=1,
            state_channel_num=16
        )

        dataset_hparams = hparams.DatasetHparams(
            dataset_name='prostate',
            batch_size=14
        )

        training_hparams = hparams.TrainingHparams(
            optimizer_name='adam',
            lr=0.001,
            training_steps='800ep',
        )

        pruning_hparams = sparse_global.PruningHparams(
            pruning_strategy='sparse_global', #TODO check pruning
            pruning_fraction=0.2,
            pruning_layers_to_ignore='fc.weight',
        )

        return LotteryDesc(model_hparams, dataset_hparams, training_hparams, pruning_hparams)
