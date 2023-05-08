import numpy as np
np.random.seed(42)
import scipy.stats as ss

import pyomo.environ as pyo

from mpisppy.opt.ph import PH
import mpisppy.utils.sputils as sputils
from mpi4py import MPI

""""
A farmer is trying to decide what to plant for the next season.
    - He has wheat (1), corn (2), and sugar beets (3)
    - He can buy (y), sell (w), and produce (x,x_yield) any of these three products.
    - He must buy / produce a min. amount of wheat / corn for his cattle.
    - Quota on sugar beets alters the price after a certain threshold.
Goal: max. the net proficts from purchasing, selling, and planting crops for next season.

However, there is no guarantees on the weather. The actual yield of the crops changes depending on weather -> aka uncertain.
Here, let's consider n scenarios.
We bin a Normal distribution, draw a sample from the truncated Normal distribution from a,b & assign the 
  probability as the P(a<=X<=b)
Solve using parallelized Progressive Hedging.

Here, we use MPI-SPPY to solve => https://mpi-sppy.readthedocs.io/en/latest/examples.html

"""

class Farmer():
    def __init__(self,predicted_yield,probability):

        # add the probability of this scenario occuring
        self.probability=probability

        # predicted_yield = a randomly drawn expected yield, with a fair weather as the mean and 2stdev as the extremes
        self.crop_yield={"wheat":2.5*predicted_yield,
                        "corn":3*predicted_yield,
                        "beets":20*predicted_yield}

        # these do not change, regardless of scenario.
        self.total_acres=500
        self.planting_cost={"wheat":150,
                            "corn":230,
                            "beets":260}
        self.planting_crops=["wheat","corn","beets"]
        self.selling_price={"wheat":170,
                            "corn":150,
                            "beets_favorable":36, 
                            "beets_unfavorable":10}
        self.selling_crops=["wheat", "corn", "beets_favorable", "beets_unfavorable"]
        self.min_requirement={"wheat":200,
                              "corn":240}
        self.purchase_price={"wheat":238,
                             "corn":210}
        self.purchasing_crops=["wheat","corn"]
        self.required_crops=self.purchasing_crops
        self.beets_quota=6000

def build_deterministic_model(scenario):
                    
    # create pyomo model
    model = pyo.ConcreteModel()

    """ VARIABLES """

    # land variables [=] acres of land devoted to each crop
    model.x=pyo.Var(scenario.planting_crops, 
                    within=pyo.NonNegativeReals)

    # selling decision variables [=] tons of crop sold
    model.w=pyo.Var(scenario.selling_crops, 
                    within=pyo.NonNegativeReals)

    # purchasing decision variables [=] tons of crop purchased
    model.y=pyo.Var(scenario.purchasing_crops, 
                    within=pyo.NonNegativeReals)

    """ CONSTRAINTS """

    # now, we split objective into first / second stage varialbes for mpi-sppy
    model.planting_cost=sum(model.x[planted_crop]*scenario.planting_cost[planted_crop] for planted_crop in scenario.planting_crops)
    model.selling_cost=sum(model.w[sold_crop]*scenario.selling_price[sold_crop] for sold_crop in scenario.selling_crops)
    model.puchasing_cost=sum(model.y[purchased_crop]*scenario.purchase_price[purchased_crop] for purchased_crop in scenario.purchasing_crops)

    model.obj=pyo.Objective( expr= model.planting_cost - model.selling_cost + model.puchasing_cost )

    # total acres allocated cannot exceed total available acreas
    @model.Constraint()
    def total_acreage_allowed(model):
        return ( sum(model.x[planted_crop] for planted_crop in scenario.planting_crops) <= scenario.total_acres )

    # must have at least x of wheat,corn
    @model.Constraint(scenario.required_crops)
    def minimum_requirement(model, required_crop):
        return ( model.x[required_crop]*scenario.crop_yield[required_crop] + model.y[required_crop] - model.w[required_crop] \
                >= scenario.min_requirement[required_crop])
    
    @model.Constraint()
    def sugar_beet_mass_balance(model):
        return ( model.w["beets_favorable"] + model.w["beets_unfavorable"] \
                <= scenario.crop_yield["beets"]*model.x["beets"] )

    # the favorably priced beets cannot exceed 6000 (T)
    @model.Constraint()
    def sugar_beet_quota(model):
        return ( model.w["beets_favorable"] <= scenario.beets_quota )
    
    return model

def scenario_creator(scenario):

    # convert scenario num from str->int
    scenario_num=int(scenario)

    # bin N() -> draw one sample from each bin & assign appropriate probability
    mu=1
    sigma=0.1
    z=2.576 # 99.% conf.
    bins=np.linspace(start=mu+z*sigma,
                    stop=mu-z*sigma,
                    num=num_scenarios+1)
    
    # order from lowest - highest
    bins=sorted(bins)

    # generate sample from truncated N(mu=1,stdev=0.1), ranges based on scenario # / bin
    predicted_yield=ss.truncnorm.rvs(a=bins[scenario_num],b=bins[scenario_num+1])
    
    # calculate the bin's probability, normalizing to 1
    cdfs={}
    total=0
    for bin_num in range(len(bins)-1):

        # calculate P(a<x<b)
        cdf_a=ss.norm.cdf(x=bins[bin_num],loc=mu,scale=sigma)
        cdf_b=ss.norm.cdf(x=bins[bin_num+1],loc=mu,scale=sigma)
        cdfs[bin_num]=cdf_b-cdf_a

        # add this probability to running total, for normalization purposes
        total+=cdf_b-cdf_a

    # calculate the normalized probabilities for each bin
    probabilities={bin_num:cdfs[bin_num]/total for bin_num in range(len(bins)-1)}

    # create parameters / model stored in obj for this scenario
    farmer_scenario=Farmer(predicted_yield,probabilities[scenario_num])
    model=build_deterministic_model(farmer_scenario)

    # add to the root node
    sputils.attach_root_node(model, model.planting_cost, [model.x])

    # add probability of scenario occuring
    model._mpisppy_probability = probabilities[scenario_num]

    return model

if __name__=="__main__":

    # parallelization param's
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # based on total num of scenarios, we have
    num_scenarios=1000

    # generate strings to label each scenario
    all_scenarios=[str(num) for num in range(num_scenarios)]
    
    # solve using PH
    options = {
        "solvername": "gurobi_persistent",
        "PHIterLimit": 10000,
        "defaultPHrho": 1,
        "convthresh": 1e-7,
        "verbose": False,
        "display_progress": False,
        "display_timing": False,
        "iter0_solver_options": dict(),
        "iterk_solver_options": dict(),
    }
    ph = PH(
            options,
            all_scenarios,
            scenario_creator,
    )
    ph_conv,ph_obj,ph_dual = ph.ph_main()

    if rank==0:
        print("num. scenarios =", num_scenarios)
        print("PH obj =", ph_obj, '\n')