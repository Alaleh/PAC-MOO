import time
import copy
import random
import shutil
import argparse
import numpy as np
from utils import *
from benchmarks import *
from model import GaussianProcess
from singlemes import MaxvalueEntropySearch
from platypus import NSGAII, Problem, Real, Subset
from scipy.optimize import minimize as scipyminimize

# PACMOO is defined as a maximization algorithm

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='pacmoo')
    parser.add_argument('--problem_name', default="OSY", type=str, help='Benchmark name')
    parser.add_argument('--initial_points', default=6, type=int, help='Number of initial BO points')
    parser.add_argument('--iterations', default=400, type=int, help='Number of BO iterations')
    parser.add_argument('--preferences', nargs='+', type=float, help='Objective preferences (sums to 1)')
    parser.add_argument('--balance', nargs='+', type=float, help='Objective/constraint balance (sums to 1)')
    parser.add_argument('--seed', default=0, type=int, help='Random seed value')
    args = parser.parse_args()
    problem_name, initial_number, total_iterations, preferences, objective_constraint_balance, seed = args.problem_name, args.initial_points, args.iterations, args.preferences, args.balance, args.seed

    # objective_constraint_balance = [0.5, 0.5]

    paths = '.'
    if os.path.exists('./results' + '_' + problem_name + '_PAC-MOO_' + str(preferences) + '/' + str(seed) + '/'):
        shutil.rmtree('./results' + '_' + problem_name + '_PAC-MOO_' + str(preferences) + '/' + str(seed) + '/')
    os.makedirs('./results' + '_' + problem_name + '_PAC-MOO_' + str(preferences) + '/' + str(seed) + '/')

    # Objectives, Constraints, input dimensions, input bounds, real_bounds, initial point count for each problem
    objective_function = get_objective_function(problem_name)
    M, C, d, input_ranges, real_ranges = get_benchmark_description(problem_name)

    np.random.seed(seed)
    random.seed(seed)

    sample_number = 1
    bound = [0, 1]
    Fun_bounds = [bound] * d
    GPs = []
    Multiplemes = []
    GPs_C = []
    Multiplemes_C = []
    selected_xs = []

    for i in range(M):
        GPs.append(GaussianProcess(d))
    for i in range(C):
        GPs_C.append(GaussianProcess(d))
    t = []

    while len(selected_xs) < initial_number:

        x_new = [
            ((random.random() * (real_ranges[p][1] - real_ranges[p][0]) + real_ranges[p][0]) - input_ranges[p][0]) / (
                    input_ranges[p][1] - input_ranges[p][0]) for p in range(d)]
        original_x, function_vals, constraint_vals = objective_function(x_new, maximization=True)

        if any([qq < 0 for qq in constraint_vals]):
            continue

        for i in range(M):
            GPs[i].addSample(np.asarray(x_new), np.asarray(function_vals[i]))
        for i in range(C):
            GPs_C[i].addSample(np.asarray(x_new), np.asarray(constraint_vals[i]))

        if problem_name in ["OSY"]:
            vals = [-x for x in function_vals]
        else:
            raise "benchmark not implemented error"
        for x in constraint_vals:
            vals.append(x)

        write_to_file(x_new, paths, 'results' + '_' + problem_name + '_PAC-MOO_' + str(preferences) + '/' + str(
            seed) + '/' + 'Inputs_PAC-MOO_' + problem_name + '.txt')
        write_to_file(original_x, paths, 'results' + '_' + problem_name + '_PAC-MOO_' + str(preferences) + '/' + str(
            seed) + '/' + 'Original_Inputs_PAC-MOO_' + problem_name + '.txt')
        write_to_file(vals, paths, 'results' + '_' + problem_name + '_PAC-MOO_' + str(preferences) + '/' + str(
            seed) + '/' + 'Outputs_PAC-MOO_' + problem_name + '.txt')

        selected_xs.append(x_new)

    for i in range(M):
        GPs[i].fitModel()
        Multiplemes.append(MaxvalueEntropySearch(GPs[i]))
    for i in range(C):
        GPs_C[i].fitModel()
        Multiplemes_C.append(MaxvalueEntropySearch(GPs_C[i]))

    for iter_num in range(initial_number, total_iterations + initial_number):

        t1 = time.time()

        for i in range(M):
            Multiplemes[i] = MaxvalueEntropySearch(GPs[i])
            Multiplemes[i].Sampling_RFM()

        for i in range(C):
            Multiplemes_C[i] = MaxvalueEntropySearch(GPs_C[i])
            Multiplemes_C[i].Sampling_RFM()

        max_samples = []
        max_samples_constraints = []

        for j in range(sample_number):

            for i in range(M):
                Multiplemes[i].weigh_sampling()
            for i in range(C):
                Multiplemes_C[i].weigh_sampling()


            def CMO(xcmo):
                xi = np.asarray(copy.deepcopy(xcmo))
                y = [Multiplemes[i].f_regression(xi)[0][0] for i in range(len(GPs))]
                y_c = [Multiplemes_C[i].f_regression(xi)[0][0] for i in range(len(GPs_C))]
                return y, y_c


            problem = Problem(nvars=d, nobjs=M, nconstrs=C, function=CMO)
            problem.types[:] = Real(0.0, 1.0)
            problem.constraints[:] = [">=0" for qq in range(C)]
            problem.directions[:] = Problem.MAXIMIZE
            algorithm = NSGAII(problem)
            algorithm.run(1500)

            cheap_pareto_front = [list(solution.objectives) for solution in algorithm.result]
            cheap_constraints_values = [list(solution.constraints) for solution in algorithm.result]

            # this is picking the max over the pareto: best case
            maxoffunctions = [-1 * min(f) for f in list(zip(*cheap_pareto_front))]
            maxofconstraints = [-1 * min(f) for f in list(zip(*cheap_constraints_values))]
            max_samples.append(maxoffunctions)
            max_samples_constraints.append(maxofconstraints)

        def pacmoo_acquisition(pacmoo_x):
            x_t = copy.deepcopy(pacmoo_x)
            if np.prod([GPs_C[i].getmeanPrediction(x_t) >= 0 for i in range(len(GPs_C))]):
                multi_obj_acq_total = 0
                for j in range(sample_number):
                    multi_obj_acq_sample = 0
                    for i in range(M):
                        multi_obj_acq_sample += objective_constraint_balance[0] * preferences[i] * Multiplemes[
                            i].single_acq(np.asarray(x_t), max_samples[j][i])
                    for i in range(C):
                        multi_obj_acq_sample += (objective_constraint_balance[1] / C) * Multiplemes_C[i].single_acq(
                            np.asarray(x_t), max_samples_constraints[j][i])
                    multi_obj_acq_total += multi_obj_acq_sample
                return multi_obj_acq_total / sample_number
            else:
                return 10e10

        x_tries = [np.asarray([random.random() for p in range(d)]) for qq in range(1000)]
        y_tries = [pacmoo_acquisition(x) for x in x_tries]
        sorted_indecies = np.argsort(y_tries)
        i = 0
        x_best = list(x_tries[sorted_indecies[i]])

        while x_best in selected_xs:
            i = i + 1
            x_best = list(x_tries[sorted_indecies[i]])

        y_best = y_tries[sorted_indecies[i]]
        x_seed = [[random.random() for p in range(d)] for qq in range(3)]
        for x_try in x_seed:
            result = scipyminimize(pacmoo_acquisition, x0=np.asarray(x_try).reshape(1, -1), method='L-BFGS-B',
                                   bounds=Fun_bounds)
            if result.success and result.fun <= y_best and result.x not in np.asarray(GPs[0].xValues):
                x_best = result.x.tolist()
                y_best = result.fun

        # ---------------Updating and fitting the GPs-----------------

        original_x, function_vals, constraint_vals = objective_function(x_best, maximization=True)

        for i in range(M):
            GPs[i].addSample(np.asarray(x_best), function_vals[i])
            GPs[i].fitModel()
        for i in range(C):
            GPs_C[i].addSample(np.asarray(x_best), constraint_vals[i])
            GPs_C[i].fitModel()

        selected_xs.append(list(x_best))

        if problem_name in ["OSY"]:
            vals = [-x for x in function_vals]
        else:
            raise "benchmark not implemented error"
        for x in constraint_vals:
            vals.append(x)

        write_to_file(x_best, paths, 'results' + '_' + problem_name + '_PAC-MOO_' + str(preferences) + '/' + str(
            seed) + '/' + 'Inputs_PAC-MOO_' + problem_name + '.txt')
        write_to_file(original_x, paths, 'results' + '_' + problem_name + '_PAC-MOO_' + str(preferences) + '/' + str(
            seed) + '/' + 'Original_Inputs_PAC-MOO_' + problem_name + '.txt')
        write_to_file(vals, paths, 'results' + '_' + problem_name + '_PAC-MOO_' + str(preferences) + '/' + str(
            seed) + '/' + 'Outputs_PAC-MOO_' + problem_name + '.txt')

        print("Seed ", seed, ", iteration ", iter_num, " took : ", time.time() - t1)