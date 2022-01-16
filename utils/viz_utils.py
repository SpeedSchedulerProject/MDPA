import numpy as np
import matplotlib.pyplot as plt

from params import args

def visualize_executor_usage(job_dags, file_path):

    exp_completion_time = int(np.ceil(np.max([j.completion_time for j in job_dags])))

    job_durations = [
        job_dag.completion_time - job_dag.start_time for job_dag in job_dags
    ]

    executor_occupation = np.zeros(exp_completion_time)
    executor_limit = np.ones(exp_completion_time) * args.exec_cap

    num_jobs_in_system = np.zeros(exp_completion_time)

    for job_dag in job_dags:
        for node in job_dag.nodes:
            for task in node.tasks:
                executor_occupation[int(task.start_time): int(task.finish_time)] += 1
        num_jobs_in_system[int(job_dag.start_time): int(job_dag.completion_time)] += 1

    executor_usage = np.sum(executor_occupation) / np.sum(executor_limit)

    fig = plt.figure()
    plt.subplot(2, 1, 1)
    # plt.plot(executor_occupation)
    # plt.fill_between(range(len(executor_occupation)), 0,
    #                  executor_occupation)
    plt.plot(moving_average(executor_occupation, 10000))
    plt.ylabel('Number of busy executors')
    plt.title(
        'Executor usage: {}'.format(executor_usage) + 
        '\n' +  
        'average completion time: {}'.format(np.mean(job_durations))
    )

    plt.subplot(2, 1, 2)
    plt.plot(num_jobs_in_system)
    plt.xlabel('Time (milliseconds)')
    plt.ylabel('Number of jobs in the system')

    fig.savefig(file_path)
    plt.close(fig)

    return executor_usage, np.mean(job_durations)


def moving_average(x, N):
    return np.convolve(x, np.ones((N,)) / N, mode='valid')
