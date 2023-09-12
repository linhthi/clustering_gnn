from nets.GAE import GAE

def GAE_Net(net_params):
    return GAE(net_params)

def cluster_model(model_name, net_params):
    model = {
        'gae': GAE_Net,
    }
    return model[model_name](net_params)