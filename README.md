# various_pyTorch_network_structure
collection of pyTorch network structure


# senet_origin.py

https://blog.csdn.net/weixin_36474809/article/details/89715056

definition of various se_resnet
and pretrained model included


# load_pretrained_model
load pretrained model from local

        if backbone == 'resnet101':
            model = models.resnet101(pretrained=False)
            print('load pretrained model...')
            model.load_state_dict(torch.load('./resnet101-5d3b4d8f.pth'))
        elif backbone == 'resnet50':
            model = models.resnet50(pretrained=False)
            print('load pretrained model...')
            model.load_state_dict(torch.load('./resnet50-5d3b4d8f.pth'))

