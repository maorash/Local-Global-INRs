from . import incode
from . import lg_incode


model_dict = {'incode': incode,
              'lc_incode': lg_incode,
              'lg_incode': lg_incode}


class INR():
    def __init__(self, name):
        self.name = name
        self.model = model_dict[name]

    def run(self, *args, **kwargs):
        if self.name == 'lc_incode':
            return lg_incode.LCINCODE(*args, **kwargs)
        if self.name == 'lg_incode':
            return lg_incode.LGINCODE(*args, **kwargs)

        return self.model.INR(*args, **kwargs)
