from torch import nn

class Model(nn.Module):
    def __init__(self, kwargs):
        super(Model, self).__init__()
        model_name = kwargs['model_name']
        exec(f"from classify.timm.models.torch_models.{model_name} import Model")
        self.model = eval("Model")(kwargs)#num_classes = kwargs.get('num_classes',1000)


    def forward(self,x):
        feature = self.model(x)
        return feature