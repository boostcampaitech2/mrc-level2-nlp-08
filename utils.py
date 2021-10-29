def send_along(func, sent_along):
    def inner(*args, **kwargs):
        return func(sent_along, *args, **kwargs)

    return inner
