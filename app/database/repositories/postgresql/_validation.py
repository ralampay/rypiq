def positive_limit(limit):
    if type(limit) is not int or limit < 1:
        raise ValueError('Limit must be a positive integer')
