import json

from .dsl import CONFIG_GROUPS, CURRENT_CONFIG_GROUP, GridParams

from . import tom_linear_gaussian, figure8, images


def get_config_group(dataset):
    for group, group_data in CONFIG_GROUPS.items():
        if dataset in group_data["datasets"]:
            return group

    assert False, f"Dataset `{dataset}' not found"


def get_datasets():
    result = []
    for items in CONFIG_GROUPS.values():
        result += items["datasets"]
    return result


def get_models():
    result = []
    for items in CONFIG_GROUPS.values():
        result += list(items["model_configs"])
    return result


def get_base_config(dataset):
    return CONFIG_GROUPS[get_config_group(dataset)]["base_config"](dataset)


def get_model_config(dataset, model):
    group = CONFIG_GROUPS[get_config_group(dataset)]
    return group["model_configs"][model](dataset, model)


def get_config(dataset, model):
    config = {
        **get_base_config(dataset),
        **get_model_config(dataset, model)
    }

    return config


def expand_grid_generator(config):
    if not config:
        yield {}
        return

    items = list(config.items())
    first_key, first_val = items[0]
    rest = dict(items[1:])

    for config in expand_grid_generator(rest):
        if isinstance(first_val, GridParams):
            for val in first_val:
                yield {
                    first_key: val,
                    **config
                }

        elif isinstance(first_val, dict):
            for sub_config in expand_grid_generator(first_val):
                yield {
                    first_key: sub_config,
                    **config
                }

        else:
            yield {
                first_key: first_val,
                **config
            }


def expand_grid(config):
    return list(expand_grid_generator(config))
