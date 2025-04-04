#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 14:12:47 2025

@author: bertrand.p
"""
import numpy as np
from copy import deepcopy

class random_weights():
    '''
    an instance of a center for GS / LS algorithm
    '''
    def __init__(self, a, f, g):
        self.weights = deepcopy(a)
        # function to optimize
        self.f = f
        self.norm = f(self.weights)
        # performance evaluation function
        self.g = g
        self.performance = g(self.weights)


class serie_random_weights:
    def __init__(self, tolerance:float, center:random_weights, multiple_dimension:bool=False):
        # Flag to identify regime
        self.cs = False
        # minimal performance
        self.tolerance = tolerance
        # current best center and CS center
        self.min_weight = center
        # current GS center
        self.last_center_gs = center
        # variance used in simulations
        self.scale = None
        self.multiple_dimension = multiple_dimension
        # current number of steps
        self.nb_steps = 1
        self.K = len(self.center().weights)
        self.n = 1
        self.C = 2.1
         
    def center(self):
        if self.cs:
            return(self.min_weight)
        else:
            return(self.last_center_gs)
    
    def simul_raw_gaussienne(self, alpha):
        self.scale = np.round(abs(alpha)/3,4)
        if self.cs:
            self.scale = self.scale/(np.sqrt(self.n*self.K))
        V = np.random.normal(loc=alpha, scale=self.scale, size=1)
        V = np.float32(V)
        return(V)

    def simul_coordinate(self, alpha):
        """
        Simulates and compares result to existing center to apply control
        """
        V = self.simul_raw_gaussienne(alpha)
        if abs(V-alpha) > self.C*alpha:
            return(alpha)
        else:
            return(V)
    
    def extend(self):
        """
        Update centers alternating GS / CS
        """
        # Current center for simulations
        centered_weights = self.center()
        a = deepcopy(centered_weights.weights)
        self.n = 1
        if not self.multiple_dimension:
            for idx, weight in enumerate(a):
                sim = self.simul_coordinate(weight)
                a[idx] = sim
        if self.multiple_dimension:
            # Iterating through the layers
            for layer_weights in reversed(a):            
                if len(layer_weights.shape) <= 1:  # Vector case (biases)
                    for idx, weight in enumerate(layer_weights):
                        sim = self.simul_coordinate(weight)
                        layer_weights[idx] = sim
                
                elif len(layer_weights.shape) == 2:  # Matrix case (weights)
                    for i, row in enumerate(layer_weights):
                        for j, weight in enumerate(row):
                            sim = self.simul_coordinate(weight)
                            layer_weights[i, j] = sim

        next_random_weight = random_weights(a, centered_weights.f, centered_weights.g)
        if next_random_weight.performance >= self.tolerance:
            if not self.cs:
                self.last_center_gs = next_random_weight
            if next_random_weight.norm < self.min_weight.norm:
                self.min_weight = next_random_weight
        self.cs = not(self.cs)
