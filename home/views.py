from django.shortcuts import render , HttpResponse
import numpy as np
from scipy.stats import *
import matplotlib.pyplot as plt
import io
import base64
from pandas import read_excel
import pandas as pd
from openpyxl import load_workbook
from django import forms
# Create your views here.
def homepage(request):
    return render(request,"indexx.html")
def mmd(request):
   return render(request,"mmd.html")
def ggd(request):
   return render(request,"ggd.html")
def mgd(request):
   return render(request,"mgd.html")
def realdata(request):
   return render(request,"realdata.html")

def mm(request):
   lambd=float(request.GET['lambd'])
   mue=float(request.GET['mue'])
   c=float(request.GET['c'])  
   lmbd = 1 / lambd
   mue = 1 / mue
   rho = lmbd / (mue * c)
   p0 = 1 - rho

   Lq = round((rho**2) / (1 - rho), 3)
   Wq = round((Lq / lmbd), 3)
   Ws = round((Wq + 1 / mue), 3)
   Ls = round((lmbd * Ws), 3)
   utilization = round((rho), 3)
   idle = p0
    
   return render(request,"result.html",{'Ls':Ls,'Lq':Lq,'Ws':Ws,'Wq':Wq,'utilization':utilization,'idle':idle})
   
def mg(request):

   c=float(request.GET['c'])
   lambd=float(request.GET['lambd'])
   dist=str(request.GET['dist'])
   min=float(request.GET['min'])
   maxx=float(request.GET['maxx']) 
   lmbd = 1 / lambd

   if dist == "Normal Distribution":
        mue = 1 / min
        var = maxx
        rho = lmbd / (mue * c)
        lq = (((lmbd ** 2) * var) + rho ** 2)
        Lq = round(lq / (2 * (1 - rho)), 3)

   elif dist == "Uniform Distribution":
        mu = (min + maxx) / 2
        mue = 1 / mu
        rho = lmbd / (mue * c)
        var_sq = ((maxx - min) ** 2) / 12
        lq = (((lmbd ** 2) * var_sq) + rho ** 2)
        Lq = round(lq / (2 * (1 - rho)), 3)

   elif dist == "Gamma Distribution":
        mu = min * maxx
        mue = 1 / mu
        rho = lmbd / (mue * c)
        var_g = min * (maxx ** 2)
        lq = (((lmbd ** 2) * var_g) + rho ** 2)
        Lq = round(lq / (2 * (1 - rho)), 3)
        #Lq = round((rho ** 2) * (min_mean_shape * (max_var_scale ** 2) + 1) / (2 * (1 - rho)), 3)

   else:
        raise ValueError("Invalid service distribution")

   Wq = round((Lq / lmbd), 3)
   Ws = round((Wq + (1 / mue)), 3)
   Ls = round((lmbd * Ws), 3)
   utilization = round((rho), 3)

   return render(request,"result.html",{'Ls':Ls,'Lq':Lq,'Ws':Ws,'Wq':Wq,'utilization':utilization})

def gg(request):

   c=float(request.GET['c'])
   dista=str(request.GET['dista'])
   dists=str(request.GET['dists'])
   minarr=float(request.GET['minarr'])
   maxarr=float(request.GET['maxarr'])
   minser=float(request.GET['minser'])
   maxser=float(request.GET['maxser']) 
   global lmbd, rho, Ca, Cs
   if dista == "Normal Distribution":
        lmbd = 1 /  minarr
        var_a = maxarr
        Ca = var_a / ((1 / lmbd) ** 2)

   elif dista == "Uniform Distribution":
        lambd = (minarr + maxarr) / 2
        lmbd = 1 / lambd
        var_sq_a = ((maxarr - minarr) ** 2) / 12
        Ca = var_sq_a / ((1 / lmbd) ** 2)

   elif dista == "Gamma Distribution":
        lambd = minarr * maxarr
        lmbd = 1 / lambd
        var_sqr_a = minarr * (maxarr ** 2)
        Ca = var_sqr_a / ((1 / lmbd) ** 2)

   else:
        raise ValueError("Invalid service distribution")

    # Service (mu)
   if dists == "Normal Distribution":
        mue = 1 /minser
        var_s =  maxser
        rho = lmbd / (mue * c)
        Cs = var_s / ((1 / mue) ** 2)

   elif dists == "Uniform Distribution":
        mu = (minser + maxser) / 2
        mue = 1 / mu
        var_sq_s = ((maxarr -minser) ** 2) / 12
        rho = lmbd / (mue * c)
        Cs = var_sq_s / ((1 / mue) ** 2)

   elif dists == "Gamma Distribution":
        mu = minser *  maxser
        mue = 1 / mu
        var_sqr_s = minser * ( maxser ** 2)
        Cs = var_sqr_s / ((1 / lmbd) ** 2)

   else:
        raise ValueError("Invalid service distribution")

   lq = ((rho ** 2) * (1 + Cs)) * (Ca + ((rho ** 2) * Cs))
   Lq = round((lq / (2 * (1 - rho) * (1 + ((rho ** 2) * Cs)))), 3)
   Wq = round((Lq / lmbd), 3)
   Ws = round((Wq + (1 / mue)), 3)
   Ls = round((lmbd * Ws), 3)
   utilization = round((rho), 3)

   return render(request,"result.html",{'Ls':Ls,'Lq':Lq,'Ws':Ws,'Wq':Wq,'utilization':utilization})
def mmdfunc(request,rand=None):
      
         nos=int(request.GET['nos'])  
         arrival=float(request.GET['arrival']) 
         service=float(request.GET['service']) 
         if rand == None:
            values = []
            i = 0
            # iterate till CP == 1
            while True:
                  val = poisson.cdf(k=i, mu=arrival)
                  values.append(val)
                  if val == 1:
                     i += 1
                     break
                  i += 1
            # length of values is our random number
            rand = len(values)

         # Inter arrivals
         inter_arrival = [0]
         inter_arvl = np.round(np.random.poisson(arrival, size=rand - 1), 2)
         inter_arvl = np.abs(inter_arvl)
         for element in inter_arvl:
            inter_arrival.append(element)

         # Arrivals
         #arrival = np.cumsum(inter_arvl)
         #arrival = np.append(arrival, arrival[-1] + inter_arvl[-1])  # Append the last arrival time
         arrival = [round(sum(inter_arrival[:i + 1]), 2) for i in range(rand)]

         # Services
         service = np.round(np.random.exponential(scale=1 / service, size=rand), 2)
         service = np.abs(service)

         # End Time
         completion = np.zeros_like(arrival)  # Initialize completion times array

         # Calculate completion times based on service times and number of servers
         for i in range(nos):
            completion[i] = arrival[i] + service[i]

         for i in range(nos, rand):
            completion[i] = max(completion[i - nos], arrival[i]) + service[i]

         # Calculate waiting times and turnaround times
         turnaround_time = abs(completion - arrival)
         waiting_time = abs(completion - arrival - service)
         completion=np.insert(completion,0,0)
         start=completion[:-1]
         response_time = abs(start - arrival)
         # turnaround_time = completion - arrival
         # waiting_time = completion - arrival - service
         # completion=np.insert(completion,0,0)
         # start=completion[:-1]
         # response_time = start - arrival

        

         # Calculate average waiting and turnaround times
         plt.plot(values,label = "Arrival")
         plt.plot(service,label = "Service")
         plt.xlabel('x - axis')
         plt.ylabel('y - axis')
         plt.title('Simulation')
         plt.legend()
         buffer1 = io.BytesIO()
         plt.savefig(buffer1, format='png')
         buffer1.seek(0)
         plot_data1 = base64.b64encode(buffer1.read()).decode() # Encode the plot as a base64 string
         plt.close()
      
         inter_arrival_mean = round(np.mean(inter_arrival), 3)
         arrival_mean = round(np.mean(arrival), 3)
         service_mean = round(np.mean(service), 3)
         avg_waiting_time = round(np.mean(np.maximum(waiting_time, 0)), 3)
         avg_turnaround_time = round(np.mean(turnaround_time), 3)
         avg_response_time = round(np.mean(response_time), 3)
         return render(request,"resultd.html",{'arrival_mean':arrival_mean, 'service_mean':service_mean, 'avg_turnaround_time':avg_turnaround_time, 'avg_waiting_time'
                                               :avg_waiting_time, 'avg_response_time':avg_response_time,'inter_arrival_mean'
                                               :inter_arrival_mean,'plot_data1': plot_data1, 'arrival':arrival,'service':service,'start': start,'completion':completion,'turnaround_time':turnaround_time,
                                               'waiting_time':waiting_time,'response_time':response_time})
            
def mgdfunc(request,rand=None):
         
         dist=str(request.GET['dist'])
         nos=int(request.GET['nos'])  
         arrival=float(request.GET['arrival']) 
         min=float(request.GET['min'])
         maxx=float(request.GET['maxx'])

         if rand == None:
            values = []
            i = 0
            # iterate till CP == 1
            while True:
                  val = poisson.cdf(k=i, mu=arrival)
                  values.append(val)
                  if val == 1:
                     i += 1
                     break
                  i += 1
            # length of values is our random number
            rand = len(values)

         # Inter arrivals
         inter_arrival = [0]
         inter_arvl = np.round(np.random.poisson(arrival, size=rand - 1), 2)
         inter_arvl = np.abs(inter_arvl)
         for element in inter_arvl:
            inter_arrival.append(element)

         # Arrivals
         #arrival = np.cumsum(inter_arvl)
         #arrival = np.append(arrival, arrival[-1] + inter_arvl[-1])  # Append the last arrival time
         arrival = [round(sum(inter_arrival[:i + 1]), 2) for i in range(rand)]         
         
         if dist == "Uniform Distribution":
            service = np.round(np.random.uniform(min, maxx, size=rand), 2)
            service = np.abs(service)

         elif dist == "Normal Distribution":
            service = np.round(np.random.normal(min, np.sqrt(maxx), size=rand), 2)
            service = np.abs(service)

         elif dist == "Gamma Distribution":
            service = np.round(np.random.gamma(min, maxx, size=rand), 2)
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
         turnaround_time = (completion - arrival)
         waiting_time = (completion - arrival - service)
         completion=np.insert(completion,0,0)
         start=completion[:-1]
         response_time =(start - arrival)

         plt.plot(values,label = "Arrival")
         plt.plot(service,label = "Service")
         plt.xlabel('x - axis')
         plt.ylabel('y - axis')
         plt.title('Simulation')
         plt.legend()
         buffer1 = io.BytesIO()
         plt.savefig(buffer1, format='png')
         buffer1.seek(0)
         plot_data1 = base64.b64encode(buffer1.read()).decode() # Encode the plot as a base64 string
         plt.close()

         # Calculate average waiting and turnaround times
         start=start
         inter_arrival_mean = round(np.mean(inter_arrival), 3)
         arrival_mean = round(np.mean(arrival), 3)
         service_mean = round(np.mean(service), 3)
         avg_waiting_time = round(np.mean(np.maximum(waiting_time, 0)), 3)
         avg_turnaround_time = round(np.mean(turnaround_time), 3)
         avg_response_time = round(np.mean(response_time), 3)

         return render(request,"resultd.html",{'arrival_mean':arrival_mean, 'service_mean':service_mean, 'avg_turnaround_time':avg_turnaround_time, 'avg_waiting_time'
                                               :avg_waiting_time, 'avg_response_time':avg_response_time,'inter_arrival_mean'
                                               :inter_arrival_mean,'plot_data1': plot_data1, 'arrival':arrival,'service':service,' start': start,'completion':completion,'turnaround_time':turnaround_time,
                                               'waiting_time':waiting_time,'response_time':response_time})
            

def ggdfunc(request,rand=None):
   
   disti=str(request.GET['disti'])
   dist=str(request.GET['dist'])
   nos=int(request.GET['nos'])  
   min=float(request.GET['min'])
   maxx=float(request.GET['maxx'])
   mini=float(request.GET['mini'])
   maxxi=float(request.GET['maxxi'])

# CP (if random is not given / to find random numbers)
   if disti == "Uniform Distribution":
        if rand == None:
            values = []
            i = 0
            # iterate till CP == 1
            while True:
                val = uniform.cdf(i, loc=mini, scale=maxxi - mini)
                values.append(val)
                if val == 1:
                    i += 1
                    break
                i += 1
            # length of values is our random number
            rand = len(values)
    
     # Inter-Arrival
        inter_arrival = [0]
        inter_arvl = np.round(np.random.uniform(mini, maxxi, size=rand - 1), 2)
        inter_arvl = np.abs(inter_arvl)
        for element in inter_arvl:
            inter_arrival.append(element)

   if disti == "Normal Distribution":
        if rand == None:
            values = []
            i = 0
            while True:
                val = norm.cdf(i, loc=mini, scale=(np.sqrt(maxxi)))
                values.append(val)
                if val == 1:
                    i += 1
                    break
                i += 1
            rand = len(values)

      # Inter-Arrival
        inter_arrival = [0]
        inter_arvl = np.round(np.random.normal(mini, np.sqrt(maxxi), size=rand - 1), 2)
        inter_arvl = np.abs(inter_arvl)
        for element in inter_arvl:
            inter_arrival.append(element)

   if disti == "Gamma Distribution":
        if rand == None:
            values = []
            i = 0
            while True:
                val = gamma.cdf(i, loc=mini, scale=maxxi)
                values.append(val)
                if val == 1:
                    i += 1
                    break
                i += 1
            rand = len(values)

    # Inter-Arrival
        inter_arrival = [0]
        inter_arvl = np.round(np.random.gamma(mini, maxxi, size=rand - 1), 2)
        inter_arvl = np.abs(inter_arvl)
        for element in inter_arvl:
            inter_arrival.append(element)

    # Arrivals
    #arrival = np.cumsum(inter_arrival)
    #arrival = np.append(arrival, arrival[-1] + inter_arrival[-1])  # Append the last arrival time
   arrival = [round(sum(inter_arrival[:i + 1]), 2) for i in range(rand)]

    # Service
   if dist == "Uniform Distribution":
        service = np.round(np.random.uniform(min, maxx, size=rand), 2)
        service = np.abs(service)

   elif dist == "Normal Distribution":
        service = np.round(np.random.normal(min, np.sqrt(maxx), size=rand), 2)
        service = np.abs(service)

   elif dist == "Gamma Distribution":
        service = np.round(np.random.gamma(min, maxx, size=rand), 2)
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
   turnaround_time = (completion - arrival)
   waiting_time = (completion - arrival - service)
   completion=np.insert(completion,0,0)
   start=completion[:-1]
   response_time =(start - arrival)

   plt.plot(values,label = "Arrival")
   plt.plot(service,label = "Service")
   plt.xlabel('x - axis')
   plt.ylabel('y - axis')
   plt.title('Simulation')
   plt.legend()
   buffer1 = io.BytesIO()
   plt.savefig(buffer1, format='png')
   buffer1.seek(0)
   plot_data1 = base64.b64encode(buffer1.read()).decode() # Encode the plot as a base64 string
   plt.close()

    # Calculate average waiting and turnaround times
   inter_arrival_mean = round(np.mean(inter_arrival), 3)
   arrival_mean = round(np.mean(arrival), 3)
   service_mean = round(np.mean(service), 3)
   avg_waiting_time = round(np.mean(np.maximum(waiting_time, 0)), 3)
   avg_turnaround_time = round(np.mean(turnaround_time), 3)
   avg_response_time = round(np.mean(response_time), 3)
   return render(request,"resultd.html",{'arrival_mean':arrival_mean, 'service_mean':service_mean, 'avg_turnaround_time':avg_turnaround_time, 'avg_waiting_time'
                                               :avg_waiting_time, 'avg_response_time':avg_response_time,'inter_arrival_mean'
                                               :inter_arrival_mean,'plot_data1': plot_data1, 'arrival':arrival,'service':service,'start': start,'completion':completion,'turnaround_time':turnaround_time,
                                               'waiting_time':waiting_time,'response_time':response_time})
            
def upload(request):
    return render(request, "fileupload.html")
def some(request):
   if request.method =="POST":
      a=str(request.POST['a'])
      b=str(request.POST['b'])
      file=request.FILES["myfile"]
      excel =pd.read_excel(file)
      # print(csv.head())
      arr1 = excel[[a]]
      arr2 = excel[[b]]
      # sumation =sum(arr)

      plt.plot(arr1,label = "Arrival")
      plt.plot(arr2,label = "Service")
      plt.xlabel('x - axis')
      plt.ylabel('y - axis')
      plt.title('Simulation')
      plt.legend()
      buffer1 = io.BytesIO()
      plt.savefig(buffer1, format='png')
      buffer1.seek(0)
      plot_data1 = base64.b64encode(buffer1.read()).decode() # Encode the plot as a base64 string
      plt.close()
      return render(request, "fileupload.html",{"something": True,"plot_data1":plot_data1})
   else:
      return render(request, "realdata.html")

     
















