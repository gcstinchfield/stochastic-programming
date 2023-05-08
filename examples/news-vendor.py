import pyomo.environ as pyo

import numpy as np
np.random.seed(42)

""""
This is a scenario representation of the farmer problem example.
  (section 1.1e, pg. 15-16, Birge & Louveax)

A news vendor must decide how many papers to purchase to try to sell that day.
    - Every morning, the news vendor buys x newspapers as price c / paper
    - There is a limit on how many the vendor can purchase, u
    - The vendor sells the newspapers (num=y) at price q
    - Any unsold newspapers (num=w) are returned to the publisher at price r (where r<c)
Goal: max. the net proficts from selling by deciding how many newspapers to buy.
Assume:
    (1) demand for newspapers varies everyday
    (2) the vendor cannot return to the publisher after he buys papers.
    (3) readers only want the paper for that particular day.

Assume deterministic demand.

"""

class Vendor():
    def __init__(self,num_scenarios):
        
        # price vendor buys at
        self.c=10
        
        # price vendor sells at
        self.q=25

        # price vendor sells back to publisher at
        self.r=5

        # generate scenarios
        self.num_scenarios=num_scenarios
        self.y=np.around(np.random.uniform(low=50,
                                 high=150,
                                 size=num_scenarios)) 

    def build_model(self):
        
        # create pyomo model
        model = pyo.ConcreteModel()

        """ PARAMETERS """

        # P(y)
        model.p=pyo.Param(default=self.num_scenarios)

        """ VARIABLES """

        # number of newspapers the newsvendor buys @ price c
        model.x=pyo.Var(within=pyo.NonNegativeIntegers)

        # number of unsold newspapers after demand is realized.
        model.w=pyo.Var(self.y,
                        within=pyo.NonNegativeIntegers)
        
        """ CONSTRAINTS """

        # max. E[profit] = min.(cost of papers bought) - E[profits sold papers] + E[leftover papers]
        model.obj=pyo.Objective( expr = model.x*self.c +
                                    (-1/model.p)*sum(y*self.q for y in self.y)
                                     + (-1/model.p)*sum(w*self.r for w in model.w) )
        
        # mass balance (cannot have the #sold + #leftover > #bought)
        @model.Constraint(self.y)
        def overall_newspaper_balance(model, y):
            return ( y + model.w[y] <= model.x )
        
        # calculate w
        @model.Constraint(self.y)
        def leftover_newspaper_balance(model, y):
            return ( model.w[y] == model.x - y )
        
        # add model to class obj
        self.model=model
    
    def solve_model(self):
        solver=pyo.SolverFactory("gurobi")
        self.results=solver.solve(self.model,tee=True)

if __name__=="__main__":

    vendor=Vendor(10)
    vendor.build_model()
    vendor.solve_model()
    vendor.model.display()