#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torchvision import datasets, transforms
import torch.nn.functional as F
import numpy as np


def average_kns(protos):
    """
    Returns the average of the local model protos.
    """
    for [label, proto_list] in protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            temp_protos = proto / len(proto_list)
            protos[label] = temp_protos * F.sigmoid(temp_protos)
        else:
            protos[label] = proto_list[0]

    return protos



