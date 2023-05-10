import pyomo.environ as pyo

import numpy as np
np.random.seed(42)

""""
This is a scenario representation of the news vendor problem example.
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

Here we use scenario sampling to solve the problem, rather than the explicit derivation.

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
        self.demand=np.around(np.random.uniform(low=50,
                                 high=150,
                                 size=num_scenarios)).tolist()
        
        # if we draw a certain scenario more than once, update probability of occurance.
        self.p={}
        for demand in sorted(set(self.demand)):
            self.p[demand]=self.demand.count(demand)/self.num_scenarios

    def build_model(self):
        
        # create pyomo model
        model = pyo.ConcreteModel()

        """ VARIABLES """

        # number of newspapers the newsvendor buys @ price c
        model.x=pyo.Var(within=pyo.NonNegativeIntegers)

        # of sold newspapers, based on RV demand
        model.y=pyo.Var(self.demand,
                        within=pyo.NonNegativeIntegers)

        # number of unsold newspapers after demand is realized.
        model.w=pyo.Var(self.demand,
                        within=pyo.NonNegativeIntegers)
        
        """ CONSTRAINTS """

        # max. E[profit] = min.(cost of papers bought) - E[profits sold papers] + E[leftover papers]
        model.obj=pyo.Objective( expr = model.x*self.c
                                    + sum((-self.p[demand])*model.y[demand]*self.q for demand in self.demand)
                                     + sum((-self.p[demand])*model.w[demand]*self.r for demand in self.demand) )
        
        # # of papers sold cannot exceed demand for that paper
        @model.Constraint(self.demand)
        def balance_sold_papers_rule(model,demand):
            return ( model.y[demand] <= demand )

        # mass balance (cannot have the #sold + #leftover > #bought)
        @model.Constraint(self.demand)
        def overall_newspaper_balance_rule(model,demand):
            return ( model.y[demand] + model.w[demand] <= model.x )
        
        # add model to class obj
        self.model=model
    
    def solve_model(self):
        solver=pyo.SolverFactory("gurobi")
        self.results=solver.solve(self.model,tee=True)

if __name__=="__main__":

    vendor=Vendor(10)
    vendor.build_model()
    vendor.solve_model()

    print("num. of papers to buy =", pyo.value(vendor.model.x))