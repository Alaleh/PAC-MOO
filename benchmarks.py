import copy


# We define the well-known CMO problems and widen their domains. This way the feasible region is small

def get_benchmark_description(name):
    # objectives, Constraints, input dimensions, input bounds, real_bounds for each problem
    benchmark_descriptions = {"OSY": [2, 7, 6, [[0, 15], [0, 15], [1, 7], [0, 9], [1, 7], [0, 15]],
                                      [[0, 10], [0, 10], [1, 5], [0, 6], [1, 5], [0, 10]]]}
    if name in benchmark_descriptions:
        return benchmark_descriptions[name]
    else:
        raise "Unimplemented benchmark error"

def OSY(xosy, maximization=True):
    # original OSY is defined for minimization ->
    # f values are multiplied in -1 to work with PACMOO as a maximization problem
    # all constraints >= 0
    x = copy.deepcopy(xosy)
    _, _, d, input_ranges, _ = get_benchmark_description("OSY")
    for i in range(d):
        x[i] = (x[i] * (input_ranges[i][1] - input_ranges[i][0])) + input_ranges[i][0]
    f_1 = -(25 * (x[0] - 2) ** 2 + (x[1] - 2) ** 2 + (x[2] - 1) ** 2 + (x[3] - 4) ** 2 + (x[4] - 1) ** 2)
    f_2 = x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2 + x[4] ** 2 + x[5] ** 2
    c_1 = x[0] + x[1] - 2
    c_2 = 6 - x[0] - x[1]
    c_3 = 2 - x[1] + x[0]
    c_4 = 2 - x[0] + 3 * x[1]
    c_5 = 4 - (x[2] - 3) ** 2 - x[3]
    c_6 = (x[4] - 3) ** 2 + x[5] - 4
    if not all ([input_ranges[i][0] <= x[i] <= input_ranges[i][1] for i in range(d)]):
        c_7 = -1
    else:
        c_7 = 1
    if maximization:
        f_1, f_2 = -f_1, -f_2
    return x, [f_1, f_2], [c_1, c_2, c_3, c_4, c_5, c_6, c_7]
