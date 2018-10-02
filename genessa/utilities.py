def name_parameter(parameter, default_name='k'):
    """ Get parameter name. """

    # set reaction parameters
    if type(parameter) in (tuple, list):
        value, name = parameter

    elif type(parameter) == dict:
        value = list(parameter.keys())[0]
        name = list(parameter.values())[0]

    # if only parameter value is provided, use default name
    elif type(parameter) in (int, float, np.float64, np.int64):
        value, name = parameter, default_name

    return value, name
