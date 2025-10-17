import torch.nn as nn
import torch.nn.functional as F
import torch

from foundations import hparams
from lottery.desc import LotteryDesc
from models import base
from pruning import sparse_global


class Model(base.Model):
    '''A NCA model for mnist'''

    def __init__(self, plan, initializer, outputs=10):
        super(Model, self).__init__()

        self.state_channel_num = 16
        self.fire_rate = 0.5
        hidden_size = 128
        self.input_channel_num = 1
        self.steps = 32

        # Perception
        self.perception_conv0 = nn.Conv2d(self.state_channel_num, self.state_channel_num, kernel_size=3, stride=1, padding=1, groups=self.state_channel_num, padding_mode="reflect")
        self.perception_conv1 = nn.Conv2d(self.state_channel_num, self.state_channel_num, kernel_size=3, stride=1, padding=1, groups=self.state_channel_num, padding_mode="reflect")

        # Update
        self.fc0 = nn.Linear(self.state_channel_num*3, hidden_size) #*3 because we have 2 convolutions and 1 identity
        self.bn = nn.BatchNorm2d(hidden_size, track_running_stats=False) #running stats false to avoid same values for each nca step
        self.fc1 = nn.Linear(hidden_size, self.state_channel_num)

        # Classifier
        self.classifier_fc0 = nn.Linear(self.state_channel_num, hidden_size)
        self.classifier_fc1 = nn.Linear(hidden_size, outputs)

        #TODO why should init fc1 with zeros?

        self.criterion = nn.CrossEntropyLoss()

        self.apply(initializer) #TODO how does initializer work?

    
    def perceive(self, x):
        CHANNEL_DIM=1
        z1 = self.perception_conv0(x)
        z2 = self.perception_conv1(x)
        y = torch.cat((x,z1,z2),CHANNEL_DIM)
        #print("-- z1, z2:" + str(z1.shape) + str(z2.shape))
        return y
    
    def update(self, x):
        #print("-- update input shape:" + str(x.shape))
        dx = self.perceive(x)
        #print("-- after perceive shape:" + str(dx.shape))
        dx = dx.transpose(1, 3)
        #print("-- after transpose shape:" + str(dx.shape))
        dx = self.fc0(dx)
        #print("-- after fc0 shape:" + str(dx.shape))
        dx = dx.transpose(1,3)
        dx = self.bn(dx) # batch norm against gradient vanishing / explosion
        dx = dx.transpose(1,3)
        dx = F.relu(dx)
        dx = self.fc1(dx)
        #print("-- after fc1 shape:" + str(dx.shape))
        dx = dx.transpose(1, 3)
        #print("-- after transpose shape:" + str(dx.shape))

        if self.fire_rate < 1.0:
            with torch.no_grad():
                stochastic = torch.rand([dx.size(0),1,dx.size(2),dx.size(3)], device=dx.device)<self.fire_rate
            dx = dx * stochastic

        x = x + dx
        return x
    
    def forward(self, x):
        # Image padding to increase the number of channels
        CHANNEL_DIM = 1
        if x.shape[CHANNEL_DIM] < self.state_channel_num:
            pad_channels = self.state_channel_num - x.shape[CHANNEL_DIM]
            zeros = torch.zeros(x.shape[0], pad_channels, x.shape[2], x.shape[3], device=x.device)
            x = torch.cat([x, zeros], dim=CHANNEL_DIM)
        
        for step in range(self.steps):
            x_updated = self.update(x)
            #print("-- x_updated:" + str(x_updated.shape))
            #print("-- x:" + str(x.shape))
            # keep input image, only update hidden state channels
            x = torch.concat((x[:, :self.input_channel_num, ...], x_updated[:, self.input_channel_num:, ...]), CHANNEL_DIM)
        
        # Average Pooling
        x = x.mean([2, 3]) # TODO use max pooling?
        #print("-- after mean:" + str(x.shape))

        x = self.classifier_fc0(x)
        x = F.relu(x)
        x = self.classifier_fc1(x)

        return x
    
    @property
    def output_layer_names(self):
        #TODO what is this
        return ['classifier_fc1.weight', 'classifier_fc1.bias']

    @staticmethod
    def is_valid_model_name(model_name):
        return (model_name.startswith('mnist_nca'))

    @staticmethod
    def get_model_from_name(model_name, initializer, outputs=None):
        """The name of a model is mnist_nca."""

        outputs = outputs or 10

        if not Model.is_valid_model_name(model_name):
            raise ValueError('Invalid model name: {}'.format(model_name))

        parts = model_name.split('_')
        if len(parts) > 2 and parts[2].isdigit():
            outputs = int(parts[2])

        plan = []
        return Model(plan, initializer, outputs)

    @property
    def loss_criterion(self):
        return self.criterion

    @staticmethod
    def default_hparams():
        model_hparams = hparams.ModelHparams(
            model_name='mnist_nca',
            model_init='kaiming_normal',
            batchnorm_init='uniform'
        )

        dataset_hparams = hparams.DatasetHparams(
            dataset_name='mnist',
            batch_size=128
        )

        training_hparams = hparams.TrainingHparams(
            optimizer_name='adam',
            lr=0.0004,
            training_steps='32ep',
        )

        pruning_hparams = sparse_global.PruningHparams(
            pruning_strategy='sparse_global', #TODO check pruning
            pruning_fraction=0.2,
            pruning_layers_to_ignore='fc.weight',
        )

        return LotteryDesc(model_hparams, dataset_hparams, training_hparams, pruning_hparams)
