import numpy as np
from scipy.stats import *


# FOR QUEUING MODEL
# M/M/C
def mmc_queue(lambd, mu, c):
    lmbd = 1 / lambd
    mue = 1 / mu
    rho = lmbd / (mue * c)
    p0 = 1 - rho

    Lq = round((rho**2) / (1 - rho), 3)
    Wq = round((Lq / lmbd), 3)
    Ws = round((Wq + 1 / mue), 3)
    Ls = round((lmbd * Ws), 3)
    utilization = round((rho), 3)
    idle = p0
    return Lq, Wq, Ws, Ls, utilization, idle


# M/G/C
def mgc_queue(lambd, c, general_distribution, min_mean_shape, max_var_scale):
    lmbd = 1 / lambd

    if general_distribution == "Normal Distribution":
        mue = 1 / min_mean_shape
        var = max_var_scale
        rho = lmbd / (mue * c)
        lq = (((lmbd ** 2) * var) + rho ** 2)
        Lq = round(lq / (2 * (1 - rho)), 3)

    elif general_distribution == "Uniform Distribution":
        mu = (min_mean_shape + max_var_scale) / 2
        mue = 1 / mu
        rho = lmbd / (mue * c)
        var_sq = ((max_var_scale - min_mean_shape) ** 2) / 12
        lq = (((lmbd ** 2) * var_sq) + rho ** 2)
        Lq = round(lq / (2 * (1 - rho)), 3)

    elif general_distribution == "Gamma Distribution":
        mu = min_mean_shape * max_var_scale
        mue = 1 / mu
        rho = lmbd / (mue * c)
        var_g = min_mean_shape * (max_var_scale ** 2)
        lq = (((lmbd ** 2) * var_g) + rho ** 2)
        Lq = round(lq / (2 * (1 - rho)), 3)
        #Lq = round((rho ** 2) * (min_mean_shape * (max_var_scale ** 2) + 1) / (2 * (1 - rho)), 3)

    else:
        raise ValueError("Invalid service distribution")

    Wq = round((Lq / lmbd), 3)
    Ws = round((Wq + (1 / mue)), 3)
    Ls = round((lmbd * Ws), 3)
    utilization = round((rho), 3)

    return Lq, Wq, Ws, Ls, utilization

# G/G/C
def ggc_queue(arrival_distribution, service_distribution, min_mean_shape_arvl, min_mean_shape_srvc, max_var_scale_arvl, max_var_scale_srvc, c):

    # Arrival (Lambda)
    global lmbd, rho, Ca, Cs
    if arrival_distribution == "Normal Distribution":
        lmbd = 1 / min_mean_shape_arvl
        var_a = max_var_scale_arvl
        Ca = var_a / ((1 / lmbd) ** 2)

    elif arrival_distribution == "Uniform Distribution":
        lambd = (min_mean_shape_arvl + max_var_scale_arvl) / 2
        lmbd = 1 / lambd
        var_sq_a = ((max_var_scale_arvl - min_mean_shape_arvl) ** 2) / 12
        Ca = var_sq_a / ((1 / lmbd) ** 2)

    elif arrival_distribution == "Gamma Distribution":
        lambd = min_mean_shape_arvl * max_var_scale_arvl
        lmbd = 1 / lambd
        var_sqr_a = min_mean_shape_arvl * (max_var_scale_arvl ** 2)
        Ca = var_sqr_a / ((1 / lmbd) ** 2)

    else:
        raise ValueError("Invalid service distribution")

    # Service (mu)
    if service_distribution == "Normal Distribution":
        mue = 1 / min_mean_shape_srvc
        var_s = max_var_scale_srvc
        rho = lmbd / (mue * c)
        Cs = var_s / ((1 / mue) ** 2)

    elif service_distribution == "Uniform Distribution":
        mu = (min_mean_shape_srvc + max_var_scale_srvc) / 2
        mue = 1 / mu
        var_sq_s = ((max_var_scale_srvc - min_mean_shape_srvc) ** 2) / 12
        rho = lmbd / (mue * c)
        Cs = var_sq_s / ((1 / mue) ** 2)

    elif service_distribution == "Gamma Distribution":
        mu = min_mean_shape_srvc * max_var_scale_srvc
        mue = 1 / mu
        var_sqr_s = min_mean_shape_srvc * (max_var_scale_srvc ** 2)
        Cs = var_sqr_s / ((1 / lmbd) ** 2)

    else:
        raise ValueError("Invalid service distribution")

    lq = ((rho ** 2) * (1 + Cs)) * (Ca + ((rho ** 2) * Cs))
    Lq = round((lq / (2 * (1 - rho) * (1 + ((rho ** 2) * Cs)))), 3)
    Wq = round((Lq / lmbd), 3)
    Ws = round((Wq + (1 / mue)), 3)
    Ls = round((lmbd * Ws), 3)
    utilization = round((rho), 3)

    return Lq, Wq, Ws, Ls, utilization

'''
Lq, Wq, Ws, Ls, utilization = ggc_queue("Normal Distribution", "Normal Distribution", 10, 8, 20, 25, 1)
print (Lq, Wq, Ws, Ls, utilization)
'''

# FOR SIMULATION
# M/M/C
def mmc_simulation(arrival_rate, service_rate, nos, rand=None):

    # CP (if random is not given / to find random numbers)
    if rand == None:
        values = []
        i = 0
        # iterate till CP == 1
        while True:
            val = poisson.cdf(k=i, mu=arrival_rate)
            values.append(val)
            if val == 1:
                i += 1
                break
            i += 1
        # length of values is our random number
        rand = len(values)

    # Inter arrivals
    inter_arrival = [0]
    inter_arvl = np.round(np.random.poisson(arrival_rate, size=rand - 1), 2)
    inter_arvl = np.abs(inter_arvl)
    for element in inter_arvl:
        inter_arrival.append(element)

    # Arrivals
    #arrival = np.cumsum(inter_arvl)
    #arrival = np.append(arrival, arrival[-1] + inter_arvl[-1])  # Append the last arrival time
    arrival = [round(sum(inter_arrival[:i + 1]), 2) for i in range(rand)]

    # Services
    service = np.round(np.random.exponential(scale=1 / service_rate, size=rand), 2)
    service = np.abs(service)

    # End Time
    completion = np.zeros_like(arrival)  # Initialize completion times array

    # Calculate completion times based on service times and number of servers
    for i in range(nos):
        completion[i] = arrival[i] + service[i]

    for i in range(nos, rand):
        completion[i] = max(completion[i - nos], arrival[i]) + service[i]

    # Calculate waiting times and turnaround times
    turnaround_time = (completion - arrival)
    waiting_time = (completion - arrival - service)
    completion=np.insert(completion,0,0)
    start=completion[:-1]
    response_time =(start - arrival)

    # Calculate average waiting and turnaround times
    inter_arrival_mean = round(np.mean(inter_arrival), 3)
    arrival_mean = round(np.mean(arrival), 3)
    service_mean = round(np.mean(service), 3)
    avg_waiting_time = round(np.mean(np.maximum(waiting_time, 0)), 3)
    avg_turnaround_time = round(np.mean(turnaround_time), 3)
    avg_response_time = round(np.mean(response_time), 3)

    return arrival_mean, service_mean, avg_turnaround_time, avg_waiting_time, avg_response_time

# M/G/C
def mgc_simulation(arrival_rate, general_distribution, min_mean_shape, max_var_scale, nos, rand=None):

    # CP (if random is not given / to find random numbers)
    if rand == None:
        values = []
        i = 0
        # iterate till CP == 1
        while True:
            val = poisson.cdf(k=i, mu=arrival_rate)
            values.append(val)
            if val == 1:
                i += 1
                break
            i += 1
        # length of values is our random number
        rand = len(values)

    # Inter arrivals
    inter_arrival = [0]
    inter_arvl = np.round(np.random.poisson(arrival_rate, size=rand - 1), 2)
    inter_arvl = np.abs(inter_arvl)
    for element in inter_arvl:
        inter_arrival.append(element)

    # Arrivals
    #arrival = np.cumsum(inter_arvl)
    #arrival = np.append(arrival, arrival[-1] + inter_arvl[-1])  # Append the last arrival time
    arrival = [round(sum(inter_arrival[:i + 1]), 2) for i in range(rand)]

    # Service
    if general_distribution == "Uniform Distribution":
        service = np.round(np.random.uniform(min_mean_shape, max_var_scale, size=rand), 2)
        service = np.abs(service)

    elif general_distribution == "Normal Distribution":
        service = np.round(np.random.normal(min_mean_shape, np.sqrt(max_var_scale), size=rand), 2)
        service = np.abs(service)

    elif general_distribution == "Gamma Distribution":
        service = np.round(np.random.gamma(min_mean_shape, max_var_scale, size=rand), 2)
        service = np.abs(service)

    else:
        raise ValueError("Invalid service distribution")

    # End Time
    completion = np.zeros_like(arrival)  # Initialize completion times array
    # Calculate completion times based on service times and number of servers
    for i in range(nos):
        completion[i] = arrival[i] + service[i]

    for i in range(nos, rand):
        completion[i] = max(completion[i - nos], arrival[i]) + service[i]

    # Calculate waiting times and turnaround times
    waiting_time = completion - arrival - service
    turnaround_time = completion - arrival
    response_time = completion - arrival

    # Calculate average waiting and turnaround times
    inter_arrival_mean = round(np.mean(inter_arrival), 3)
    arrival_mean = round(np.mean(arrival), 3)
    service_mean = round(np.mean(service), 3)
    avg_waiting_time = round(np.mean(np.maximum(waiting_time, 0)), 3)
    avg_turnaround_time = round(np.mean(turnaround_time), 3)
    avg_response_time = round(np.mean(response_time), 3)

    return arrival_mean, service_mean, avg_turnaround_time, avg_waiting_time, avg_response_time

# G/G/C
def ggc_simulation(arrival_distribution, service_distribution, min_mean_shape_arvl, min_mean_shape_srvc, max_var_scale_arvl, max_var_scale_srvc, nos, rand=None):

    # CP (if random is not given / to find random numbers)
    if arrival_distribution == "Uniform Distribution":
        if rand == None:
            values = []
            i = 0
            # iterate till CP == 1
            while True:
                val = uniform.cdf(i, loc=min_mean_shape_arvl, scale=max_var_scale_arvl - min_mean_shape_arvl)
                values.append(val)
                if val == 1:
                    i += 1
                    break
                i += 1
            # length of values is our random number
            rand = len(values)
    
     # Inter-Arrival
        inter_arrival = [0]
        inter_arvl = np.round(np.random.uniform(min_mean_shape_arvl, max_var_scale_arvl, size=rand - 1), 2)
        inter_arvl = np.abs(inter_arvl)
        for element in inter_arvl:
            inter_arrival.append(element)

    if arrival_distribution == "Normal Distribution":
        if rand == None:
            values = []
            i = 0
            while True:
                val = norm.cdf(i, loc=min_mean_shape_arvl, scale=(np.sqrt(max_var_scale_arvl)))
                values.append(val)
                if val == 1:
                    i += 1
                    break
                i += 1
            rand = len(values)

      # Inter-Arrival
        inter_arrival = [0]
        inter_arvl = np.round(np.random.normal(min_mean_shape_arvl, np.sqrt(max_var_scale_arvl), size=rand - 1), 2)
        inter_arvl = np.abs(inter_arvl)
        for element in inter_arvl:
            inter_arrival.append(element)

    if arrival_distribution == "Gamma Distribution":
        if rand == None:
            values = []
            i = 0
            while True:
                val = gamma.cdf(i, loc=min_mean_shape_arvl, scale=max_var_scale_arvl)
                values.append(val)
                if val == 1:
                    i += 1
                    break
                i += 1
            rand = len(values)

    # Inter-Arrival
        inter_arrival = [0]
        inter_arvl = np.round(np.random.gamma(min_mean_shape_arvl, max_var_scale_arvl, size=rand - 1), 2)
        inter_arvl = np.abs(inter_arvl)
        for element in inter_arvl:
            inter_arrival.append(element)

    # Arrivals
    #arrival = np.cumsum(inter_arrival)
    #arrival = np.append(arrival, arrival[-1] + inter_arrival[-1])  # Append the last arrival time
    arrival = [round(sum(inter_arrival[:i + 1]), 2) for i in range(rand)]

    # Service
    if service_distribution == "Uniform Distribution":
        service = np.round(np.random.uniform(min_mean_shape_srvc, max_var_scale_srvc, size=rand), 2)
        service = np.abs(service)

    elif service_distribution == "Normal Distribution":
        service = np.round(np.random.normal(min_mean_shape_srvc, np.sqrt(max_var_scale_srvc), size=rand), 2)
        service = np.abs(service)

    elif service_distribution == "Gamma Distribution":
        service = np.round(np.random.gamma(min_mean_shape_srvc, max_var_scale_srvc, size=rand), 2)
        service = np.abs(service)

    else:
        raise ValueError("Invalid service distribution")


    # End Time
    completion = np.zeros_like(arrival)  # Initialize completion times array
    # Calculate completion times based on service times and number of servers
    for i in range(nos):
        completion[i] = arrival[i] + service[i]

    for i in range(nos, rand):
        completion[i] = max(completion[i - nos], arrival[i]) + service[i]

    # Calculate waiting times and turnaround times
    waiting_time = completion - arrival - service
    turnaround_time = completion - arrival
    response_time = completion - arrival

    # Calculate average waiting and turnaround times
    inter_arrival_mean = round(np.mean(inter_arrival), 3)
    arrival_mean = round(np.mean(arrival), 3)
    service_mean = round(np.mean(service), 3)
    avg_waiting_time = round(np.mean(np.maximum(waiting_time, 0)), 3)
    avg_turnaround_time = round(np.mean(turnaround_time), 3)
    avg_response_time = round(np.mean(response_time), 3)

    return arrival_mean, service_mean, avg_turnaround_time, avg_waiting_time, avg_response_time
'''
values = ggc_simulation("Normal Distribution", "Normal Distribution", 10, 8, 20, 25, 1)
print(values)
'''