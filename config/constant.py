# coding = utf-8
import sys


class Const:
    class ConstError(PermissionError):
        pass

    class ConstDefineError(TypeError):
        pass

    def __setattr__(self, key, value):
        if key in self.__dict__:
            raise self.ConstError('Constant assigned! ')
        if not key.isupper():
            raise self.ConstDefineError('Constant not capitalized! ')
        self.__dict__[key] = value


sys.modules[__name__] = Const()

