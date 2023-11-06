import sys
import argparse
import os, signal
from time import time
from argparse import ArgumentParser
from multiprocessing import Process, Queue, cpu_count


def worker(cmd, problem, initials, iters, prefs, bals, seed, queue):
    ret_code = os.system(cmd)
    queue.put([ret_code, problem, initials, iters, prefs, bals, seed])


def main():

    # python run_pacmoo.py --problem_name OSY --initial_points 6 --iterations 100 --preferences 0.8 0.2 --balance 0.5 0.5 --seeds 5

    parser = argparse.ArgumentParser(description='parallel')
    parser.add_argument('--problem_name', default="OSY", type=str, help='Benchmark name')
    parser.add_argument('--initial_points', default=6, type=int, help='Number of initial BO points')
    parser.add_argument('--iterations', default=100, type=int, help='Number of BO iterations')
    parser.add_argument('--preferences', nargs='+', type=float, help='Objective preferences (sums to 1)')
    parser.add_argument('--balance', nargs='+', type=float, help='Objective/constraint balance (sums to 1)')
    parser.add_argument('--seeds', default=10, type=int, help='Number of random seeds')
    args = parser.parse_args()

    queue = Queue()
    n_active_process = 0
    start_time = time()
    n_process = cpu_count()
    preference_str = ""
    for i in args.preferences:
        preference_str += str(i) + ' '
    balance_str = ""
    for i in args.balance:
        balance_str += str(i) + ' '

    for seed in range(args.seeds):
        command = f'python -W ignore pacmoo.py --problem {args.problem_name} --initial_points {args.initial_points} --iterations {args.iterations} --preferences {preference_str} --balance {balance_str} --seed {seed}'
        Process(target=worker, args=(command, args.problem_name, args.initial_points, args.iterations, args.preferences, args.balance, seed, queue)).start()
        print(f'problem {args.problem_name}, preference values {args.preferences}, seed {seed} started')
        n_active_process += 1

        if n_active_process >= n_process:
            ret_code, ret_problem, ret_initial, ret_iter, ret_pref, ret_bal, ret_seed = queue.get()
            if ret_code == signal.SIGINT:
                exit()
            print(
                f'problem {args.problem_name}, preference values {args.preferences}, seed {seed} done, time: ' + '%.2fs' % (
                        time() - start_time))
            n_active_process -= 1

    for _ in range(n_active_process):
        ret_code, ret_problem, ret_initial, ret_iter, ret_pref, ret_bal, ret_seed = queue.get()
        if ret_code == signal.SIGINT:
            exit()
        print(
            f'problem {args.problem_name}, preference values {args.preferences}, seed {ret_seed} done, time: ' + '%.2fs' % (
                    time() - start_time))

    print('all experiments done, time: %.2fs' % (time() - start_time))


if __name__ == "__main__":
    main()
