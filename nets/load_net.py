from nets.GAE import GAE

def GAE(net_params):
    return GAE(net_params)

def cluster_model(model_name, net_params):
    model = {
        'GAE': GAE,
    }
    return model[model_name](net_params)