""" Base class to describe a parameter
"""
from __future__ import print_function
# from itertools import groupby

class Param(object):

    def __init__(self, name="", value=None, group="", help="", units="", **kwargs):
        self.name = name.lower()  # keep lower case (fortran-written namelist have upper case otherwise !
        self.value = value
        self.group = group.lower()
        self.help = help  # .e.g. "blabla ({units})"
        self.units = units
        if (len(kwargs) > 0):
            warnings.warn("unknown parameters to Param were ignored: "+", ".join(kwargs.keys()))
        # self.__dict__.update(kwargs)

    @property
    def key(self):
        " unique ID "
        return (self.group, self.name)

    def __eq__(self, other):
        if not isinstance(other, Param): 
            # return False
            raise TypeError("Expected Param, got: {}".format(type(other)))
        return self.key == other.key

    def __repr__(self):
        return "Param(name=%r, value=%r, group=%r)" % (self.name, self.value, self.group)

class Params(list):
    """ list of parameters
    """
    def __init__(self, *args):
        # make sure we have a list
        list.__init__(self, *args)
        for p in self:
            if not isinstance(p, Param):
                print(type(p),":",p)
                raise TypeError("Expected Param, got: {}".format(type(p)))

    def append(self, param):
        if not isinstance(param, Param):
            raise TypeError("Expected Param, got: {}".format(type(param)))
        list.append(self, param)

    def extend(self, params):
        if not isinstance(other, Params):
            raise TypeError("Expected Params, got: {}".format(type(params)))
        list.extend(self, params)

    def to_json(self, **kwargs):
        import json
        return json.dumps([vars(p) for p in self], **kwargs)

    @classmethod
    def from_json(cls, string):
        import json
        return cls([Param(**p) for p in json.loads(string)])

    # generic method to be overloaded, default to json
    parse = from_json
    format = to_json

    def write(self, filename, mode='w', **kwargs):
        with open(filename, mode) as f:
            f.write(self.format(**kwargs))

    @classmethod
    def read(cls, filename):
        with open(filename) as f:
            params = cls.parse(f.read())
        return params

    # def __repr__(self):
        # return object.__repr__()
        # return "{cls}({list})".format(cls=self.__class__.__name__, list=list.__repr__(self))
        # return "{cls}({list})".format(cls=self.__class__.__name__, list=list.__repr__(self))

    #
    # Make it easier to get/set param value, but still in the philosophy of the list structure
    #
    def filter(self, **kwargs):
        def func(p):
            res = True
            for k in kwargs:
                res = res and getattr(p, k) == kwargs[k]
            return res
        return self.__class__(filter(func, self))

    def update(self, others, extends=False):
        """ update existing parameters
        """
        for p in others:
            if p not in self and extends:
                self.append(p)
            else:
                self[self.index(p)] = p  

    def search(self, **kwargs):
        """ return one parameter matching the asked-for criteria
        will raise an error if not exactly one param is found

        >>> params.search(name='rho_i').value = 910
        >>> params.search(name='A', group="dynamics").help
        """
        ps = self.filter(**kwargs)
        assert len(ps) > 0, 'no param found'
        assert len(ps) == 1, 'several params found'
        return ps[0]

    # def groupby(self, group):
    #     """ wrapper around `itertools.groupby` 
    #     >>> for group_name, group_params in params.groupby('group'):
    #     ...     print 'Group:', group_name
    #     ...     for p in group_params:
    #     ...         print p
    #     """
    #     for group_name, group_params in groupby(params, lambda x: getattr(x, g)):
    #         yield group_name, group_params
