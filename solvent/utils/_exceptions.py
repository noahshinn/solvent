
class DataNotLoadedException(Exception):
    def __init__(self, msg: str) -> None:
        super().__init__(msg)

class InvalidFileType(Exception):
    def __init__(self, msg: str) -> None:
        super().__init__(msg)
