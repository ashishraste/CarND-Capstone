#!/usr/bin/env python

# Common utility methods class
class Helper(object):
    def __init__(self):
        pass

    @staticmethod
    def get_none_instances(num_instances):
        nones = lambda x: [None for i in range(x)]
        return nones(num_instances)