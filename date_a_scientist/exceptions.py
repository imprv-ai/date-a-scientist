class BaseException(Exception):
    def __init__(self, message):
        self.message = message


class ModelNotFoundError(BaseException):
    pass
