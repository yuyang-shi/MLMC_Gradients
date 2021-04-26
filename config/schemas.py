def get_schema(config):
    schema = {}
    schema["encoder"] = {**config["encoder"]}
    schema["decoder"] = {**config["decoder"]}

    schema["encoder"]["net_dims"] = [config["obs_dim"][0]] + config["encoder"]["net_dims"] + [config["latent_dim"]]
    schema["decoder"]["net_dims"] = [config["latent_dim"]] + config["encoder"]["net_dims"] + [config["obs_dim"][0]]

    if "prior" in config.keys():
        schema["prior"] = config["prior"]
        schema["prior"]["net_dims"] = [config["latent_dim"]]
    else:
        # default N(0, I) prior
        schema["prior"] = {"network_type": "Constant_Normal_Constant_Sigma",
                           "net_dims": [config["latent_dim"]],
                           "distribution_type": "Normal",
                           "init_mu": 0.,
                           "init_logsigma": 0.,
                           "fix_mu": True,
                           "fix_logsigma": True}
    return schema