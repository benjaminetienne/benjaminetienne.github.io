---
layout: page
title: OR-tools
---

# Optimisation


```python
import numpy as np
import itertools
from ortools.linear_solver import pywraplp
```

<h2> 0. Summary</h2>

1. [Production Problem](#section-1)
2. [Dual Problem](#section-2)
3. [Mixed Integer Programming - Warehouse allocation](#section-3)
3. [Mixed Integer Programming - Water Network](#section-4)

<a id="section-1"></a><h2>1. Linear Programming : Production Problem</h2>[Home](#home)

A biscuit factory produces cookies, cupcakes, and  brownies which are then sold at different prices.
Each type of biscuits needs some flour, sugar, chocolate, vegetal oil, and eggs.
For a given quantity of goods, the factory manager wants to maximize its revenue.

Formulation of the problem

|           |              |
| ----------|:-------------|
| $w_{i,j}$ | weight needed of ingredient $i$ to produce biscuit $j$ |
| $c_j$     | price for biscuit $j$ |
| $s_i$     | available stock for ingredient $i$ |
| $x_j$     | quantity of biscuit $j$ produced |

\begin{align*}
\max &\sum_j c_j * x_j & \\
\text{subject to:}&&\\
&\sum_j w_{i,j} * x_j \leq s_i & \forall i \\
&x_j \geq 0 & \forall j
\end{align*}



```python
BISCUITS = ['cookie', 'cupcake', 'brownie']
N_BISCUITS = len(BISCUITS)
GOODS = ['flour', 'sugar', 'chocolate', 'oil', 'eggs']
N_GOODS = len(GOODS)

# quantities
w = np.array(
    [[.1, .3, .05], # flour
     [.15, .2, .3], # sugar
     [.1, .05, .3], # chocolate
     [.05, .1, .1], # oil
     [0, 2, 3]]     # eggs
)     
c = np.array([2, 3, 5]) # selling price of biscuits
s = np.array([25, 40, 30, 15, 300])  # stock of ingredients
```

We first start by instantiating the solver :


```python
solver = pywraplp.Solver.CreateSolver('SCIP')
```

We then craete the variables 


```python
x = {}

for n, biscuit in enumerate(BISCUITS):
    x[n] = solver.NumVar(0, solver.infinity(), biscuit)
```


```python
x
```




    {0: cookie, 1: cupcake, 2: brownie}



We then add the constraints to our model:


```python
n_constraints = len(s)

for i in range(n_constraints):
    solver.Add(sum([x[j]*w[i, j] for j in range(N_BISCUITS)]) <= s[i])
```

And finally we add the objective function


```python
objective = sum([x[i]*c[i] for i in range(N_BISCUITS)])
solver.Maximize(objective)
```


```python
# Run the solver
status = solver.Solve()
if status == pywraplp.Solver.OPTIMAL:
    print('Objective value =', solver.Objective().Value())
    for j in range(N_BISCUITS):
        print(" #{} {} = {}".format(
            j, x[j].name(), x[j].solution_value())
        )
        
    print()
    print('Problem solved in %f milliseconds' % solver.wall_time())
    print('Problem solved in %d iterations' % solver.iterations())
    print('Problem solved in %d branch-and-bound nodes' % solver.nodes())
else:
    print(status)
    print('The problem does not have an optimal solution.')
```

    Objective value = 618.5185185185185
     #0 cookie = 66.66666666666664
     #1 cupcake = 44.44444444444442
     #2 brownie = 70.37037037037038
    
    Problem solved in 2102.000000 milliseconds
    Problem solved in 3 iterations
    Problem solved in 1 branch-and-bound nodes


<a id="section-2"></a><h2>2. Linear Programming : Duality</h2>[Home](#home)

The dual formulation is:

|           |              |
| ----------|:-------------|
| $w_{i,j}$ | weight needed of good $i$ to produce biscuit $j$ |
| $c_j$     | price for biscuit $j$ |
| $s_i$     | available stock for good $i$ |
| $y_i$     | quantity of good $i$ used |

\begin{align*}
\min &\sum_i s_i * y_i & \\
\text{subject to:}&&\\
&\sum_i w_{j,i} * y_i \geq c_j & \forall j \\
&y_i \geq 0 & \forall i
\end{align*}

And it's implementation:


```python
dual_model = pywraplp.Solver.CreateSolver('SCIP')
```

In the dual problem, the variables are now the ingredients 


```python
y = {}

for n, good in enumerate(GOODS):
    y[n] = dual_model.NumVar(0, solver.infinity(), good)
```


```python
y
```




    {0: flour, 1: sugar, 2: chocolate, 3: oil, 4: eggs}




```python
n_constraints = len(c)

for j in range(n_constraints):
    dual_model.Add(sum([w[i, j]*y[i] for i in range(N_GOODS)]) >= c[j])
```


```python
objective = sum([y[i]*s[i] for i in range(N_GOODS)])
dual_model.Minimize(objective)
```


```python
# Run the solver
status = dual_model.Solve()
print(status)
if status == pywraplp.Solver.OPTIMAL:
    print('Objective value =', dual_model.Objective().Value())
    for i in range(N_GOODS):
        print(" #{} {} = {}".format(
            i, y[i].name(), y[i].solution_value())
        )
else:
    print(status)
    print('The problem does not have an optimal solution.')
```

    0
    Objective value = 618.5185185185185
     #0 flour = 0.0
     #1 sugar = 11.85185185185185
     #2 chocolate = 2.2222222222222254
     #3 oil = 0.0
     #4 eggs = 0.259259259259259



```python
print('{:<12} | {:^12} | {:^12}'.format('', 'Primal model', 'Dual model'))
print(''.join(['=']*50))
print('{:<12} | {:^12} | {:^12}'.format('', 'Variable', 'Slack'))
print(''.join(['-']*50))
for j in range(N_BISCUITS):
    print('{:<12} | {:>12.2f} | {:>12.2f}'.format(
        BISCUITS[j], x[j].solution_value() , c[j] - sum([y[i].solution_value()*w[i, j] for i in range(N_GOODS)])))
print(''.join(['-']*50))
print('{:<12} | {:^12} | {:^12}'.format('', 'Slack', 'Variable'))
print(''.join(['-']*50))
for i in range(N_GOODS):
    print('{:<12} | {:>12.2f} | {:>12.2f}'.format(
        GOODS[i],  s[i] - sum([x[j].solution_value()*w[i, j] for j in range(N_BISCUITS)]), y[i].solution_value()))
```

                 | Primal model |  Dual model 
    ==================================================
                 |   Variable   |    Slack    
    --------------------------------------------------
    cookie       |        66.67 |         0.00
    cupcake      |        44.44 |         0.00
    brownie      |        70.37 |         0.00
    --------------------------------------------------
                 |    Slack     |   Variable  
    --------------------------------------------------
    flour        |         1.48 |         0.00
    sugar        |         0.00 |        11.85
    chocolate    |         0.00 |         2.22
    oil          |         0.19 |         0.00
    eggs         |         0.00 |         0.26


<a id="section-3"></a><h2>3. Mixed Integer Programming : WareHouse Problem</h2>[Home](#home)

<h3>Problem definition </h3>
<br>
We have several warehouses w and n customers to serve. We want to know which warehouse should serve which customers, given that :

- Each warehouse has a fixed capacity
- Serving a customer has a cost
    

<h3>Decision variables</h3>

- decide whether a warehouse serves a customer
     - $y_{wc}$ = 1 if warehouse w serves customer c

<h3>What are the constraints?</h3>

- the warehouse cannot serve more customers than its capacity \begin{align*} &\sum_c y_{wc} \leq capa_{w} & \forall w \\\end{align*}
- a customer must be served by exactly one warehouse \begin{align*} &\sum_w y_{wc} = 1 & \forall c \\\end{align*}

### Problem Formulation

\begin{align*}
\min &\qquad \sum_{w ,s} t_{wc} y_{wc} & \\
\text{subject to:} &&\\
&\sum_w y_{wc} = 1 & \forall c \\
&x_w, y_{wc} \in \mathbb{B} & \forall w,c
\end{align*}


```python
w_capacity = [1,4,1,3,2]  # the capacity of a warehouse

transportation_costs = np.array(
    [[20, 28, 74,  2, 46, 42,  1, 10, 93, 47], # t_{w,s}
     [24, 27, 97, 55, 96, 22,  5, 73, 35, 65],
     [11, 82, 71, 73, 59, 29, 73, 13, 63, 55],
     [25, 83, 96, 69, 83, 67, 59, 43, 85, 71],
     [30, 74, 70, 61,  4, 59, 56, 96, 46, 95]])


N_WAREHOUSES, N_STORES  = transportation_costs.shape
```


```python
# --- Create the solver
mip_model = pywraplp.Solver.CreateSolver("SAT")

# --- Create the variables
y = {}
for w, s in itertools.product(range(N_WAREHOUSES), range(N_STORES)):
    y[w, s] = mip_model.IntVar(0, 1, f"y[{w, s}]")
print("Number of variables =", mip_model.NumVariables())


# --- Create the constraints
for w in range(N_WAREHOUSES):
    mip_model.Add(sum([y[w, c] for c in range(N_STORES)]) <= w_capacity[w])
    
for c in range(N_STORES):
    mip_model.Add(sum([y[w, c] for w in range(N_WAREHOUSES)]) == 1)
    
# -- Define the objective function
objective = sum([y[i, j]*transportation_costs[i, j] for i, j in itertools.product(range(N_WAREHOUSES), range(N_STORES))])
mip_model.Minimize(objective)

print(f"Solving with {solver.SolverVersion()}")
status = mip_model.Solve()

if status == pywraplp.Solver.OPTIMAL:
    print("Objective value =", mip_model.Objective().Value())
    solutions = np.zeros((N_WAREHOUSES, N_STORES))
    for s, w in itertools.product(range(N_STORES), range(N_WAREHOUSES)):
        solutions[w,s] = y[w,s].solution_value()
        if solutions[w,s] == 1:
            print("Store {} will be served by warehouse {}".format(s,w))
    print()
    print(f"Problem solved in {mip_model.wall_time():d} milliseconds")
    print(f"Problem solved in {mip_model.iterations():d} iterations")
    print(f"Problem solved in {mip_model.nodes():d} branch-and-bound nodes")
    # print(solutions)
else:
    print("The problem does not have an optimal solution.")
```

    Number of variables = 50
    Solving with SCIP 9.0.0 [LP solver: Glop 9.10]
    Objective value = 274.0
    Store 0 will be served by warehouse 3
    Store 1 will be served by warehouse 1
    Store 2 will be served by warehouse 4
    Store 3 will be served by warehouse 0
    Store 4 will be served by warehouse 4
    Store 5 will be served by warehouse 1
    Store 6 will be served by warehouse 1
    Store 7 will be served by warehouse 2
    Store 8 will be served by warehouse 1
    Store 9 will be served by warehouse 3
    
    Problem solved in 12 milliseconds
    Problem solved in 0 iterations
    Problem solved in 0 branch-and-bound nodes


---

Let's now consider the rental prices of your warehouses: 
The rents of the warehouses are :

```python
rents = [20, 75, 18, 34, 22]
```

Your landlord for warehouse 2 is asking for a +50% increase, from 18 to 27k€ per month. The question is :

-> **Should you close store 2 or should you accept the raise?**

We will start by adding this term to our objective function : 
\begin{align*}
\min &\qquad \sum_w rent_w x_w + \sum_{w ,s} t_{ws} y_{ws} & \\
\end{align*}

<h3>Decision variables</h3>

- decide whether a warehouse serves a customer
     - $y_{wc}$ = 1 if warehouse w serves customer c
- <strong>for each warehouse, decide whether to open it
    - $x_w$ = 1 if warehouse w is open</strong>

<h3>What are the constraints?</h3>

- the warehouse cannot serve more customers than its capacity \begin{align*} &\sum_c y_{wc} \leq capa_{w} & \forall w \\\end{align*}
- a customer must be served by exactly one warehouse \begin{align*} &\sum_w y_{wc} = 1 & \forall c \\\end{align*}
- <strong>a warehouse can serve a customer only if it is open \begin{align*} &y_{wc} \leq x_w & \forall w,c \\\end{align*}</strong>

<h3>What is the objective function ?</h3>

We want to minimize all three:

- the cost of opening a warehouse
- the transportation cost between the customer and the warehouse

<h3>Problem definition </h3>


```python
# --- Create the solver
mip_model = pywraplp.Solver.CreateSolver("SAT")
rents = [20, 75, 18, 34, 22]


# --- Create the variables
y = {}
for w, s in itertools.product(range(N_WAREHOUSES), range(N_STORES)):
    y[w, s] = mip_model.IntVar(0, 1, f"y[{w, s}]")
print("Number of variables =", mip_model.NumVariables())
x = {}
for w in range(N_WAREHOUSES):
    x[w] = mip_model.IntVar(0, 1, f"x[{w}]")
print("Number of variables =", mip_model.NumVariables())



# --- Create the constraints
for w in range(N_WAREHOUSES):
    mip_model.Add(sum([y[w, c] for c in range(N_STORES)]) <= w_capacity[w])
    
for c in range(N_STORES):
    mip_model.Add(sum([y[w, c] for w in range(N_WAREHOUSES)]) == 1)
    
for w, c in itertools.product(range(N_WAREHOUSES), range(N_STORES)):
    mip_model.Add(y[w, c] <= x[w])
    
# -- Define the objective function
objective = sum([y[i, j]*transportation_costs[i, j] for i, j in itertools.product(range(N_WAREHOUSES), range(N_STORES))])
objective += sum([x[i] * rents[i] for i in range(N_WAREHOUSES)])
mip_model.Minimize(objective)

print(f"Solving with {solver.SolverVersion()}")
status = mip_model.Solve()

if status == pywraplp.Solver.OPTIMAL:
    print("Objective value with current rents =", mip_model.Objective().Value())
else:
    print("The problem does not have an optimal solution.")
```

    Number of variables = 50
    Number of variables = 55
    Solving with SCIP 9.0.0 [LP solver: Glop 9.10]
    Objective value with current rents = 443.0



```python
# --- Create the solver
mip_model = pywraplp.Solver.CreateSolver("SAT")
rents = [20, 75, 27, 34, 22]


# --- Create the variables
y = {}
for w, s in itertools.product(range(N_WAREHOUSES), range(N_STORES)):
    y[w, s] = mip_model.IntVar(0, 1, f"y[{w, s}]")
print("Number of variables =", mip_model.NumVariables())
x = {}
for w in range(N_WAREHOUSES):
    x[w] = mip_model.IntVar(0, 1, f"x[{w}]")
print("Number of variables =", mip_model.NumVariables())



# --- Create the constraints
for w in range(N_WAREHOUSES):
    mip_model.Add(sum([y[w, c] for c in range(N_STORES)]) <= w_capacity[w])
    
for c in range(N_STORES):
    mip_model.Add(sum([y[w, c] for w in range(N_WAREHOUSES)]) == 1)
    
for w, c in itertools.product(range(N_WAREHOUSES), range(N_STORES)):
    mip_model.Add(y[w, c] <= x[w])
    
# -- Define the objective function
objective = sum([y[i, j]*transportation_costs[i, j] for i, j in itertools.product(range(N_WAREHOUSES), range(N_STORES))])
objective += sum([x[i] * rents[i] for i in range(N_WAREHOUSES)])
mip_model.Minimize(objective)

print(f"Solving with {solver.SolverVersion()}")
status = mip_model.Solve()

if status == pywraplp.Solver.OPTIMAL:
    print("Objective value with new rents =", mip_model.Objective().Value())
else:
    print("The problem does not have an optimal solution.")
```

    Number of variables = 50
    Number of variables = 55
    Solving with SCIP 9.0.0 [LP solver: Glop 9.10]
    Objective value with new rents = 452.0


With the rent increase for store 2, costs have raised from 443 to 452k€. 

How about closing one store to compensate for this raise ? 

We add an additional constraint :

- the warehouse cannot serve more customers than its capacity \begin{align*} &\sum_c y_{wc} \leq capa_{w} & \forall w \\\end{align*}
- a customer must be served by exactly one warehouse \begin{align*} &\sum_w y_{wc} = 1 & \forall c \\\end{align*}
- <strong>a warehouse can serve a customer only if it is open \begin{align*} &y_{wc} \leq x_w & \forall w,c \\\end{align*}</strong>
- <strong>we can only open 4 warehouses \begin{align*} &\sum_w x_{w} = 4  \\\end{align*}</strong>


```python
# --- Create the solver
mip_model = pywraplp.Solver.CreateSolver("SAT")
rents = [20, 75, 27, 34, 22]


# --- Create the variables
y = {}
for w, s in itertools.product(range(N_WAREHOUSES), range(N_STORES)):
    y[w, s] = mip_model.IntVar(0, 1, f"y[{w, s}]")
print("Number of variables =", mip_model.NumVariables())
x = {}
for w in range(N_WAREHOUSES):
    x[w] = mip_model.IntVar(0, 1, f"x[{w}]")
print("Number of variables =", mip_model.NumVariables())



# --- Create the constraints
for w in range(N_WAREHOUSES):
    mip_model.Add(sum([y[w, c] for c in range(N_STORES)]) <= w_capacity[w])
    
for c in range(N_STORES):
    mip_model.Add(sum([y[w, c] for w in range(N_WAREHOUSES)]) == 1)
    
for w, c in itertools.product(range(N_WAREHOUSES), range(N_STORES)):
    mip_model.Add(y[w, c] <= x[w])
    
mip_model.Add(sum([x[w] for w in range(N_WAREHOUSES)]) <= 4)

# -- Define the objective function
objective = sum([y[i, j]*transportation_costs[i, j] for i, j in itertools.product(range(N_WAREHOUSES), range(N_STORES))])
objective += sum([x[i] * rents[i] for i in range(N_WAREHOUSES)])
mip_model.Minimize(objective)

print(f"Solving with {solver.SolverVersion()}")
status = mip_model.Solve()

if status == pywraplp.Solver.OPTIMAL:
    print("Objective value with new rents =", mip_model.Objective().Value())
    for i in range(N_WAREHOUSES):
        print(f"Warehouse {i} -> {'Keep' if x[i].solution_value()==1 else 'Close'}")
else:
    print("The problem does not have an optimal solution.")
```

    Number of variables = 50
    Number of variables = 55
    Solving with SCIP 9.0.0 [LP solver: Glop 9.10]
    Objective value with new rents = 455.0
    Warehouse 0 -> Keep
    Warehouse 1 -> Keep
    Warehouse 2 -> Close
    Warehouse 3 -> Keep
    Warehouse 4 -> Keep


Shutting down warehouse 2 would be 3k more expensive than to accept the rent raise.

<a id="section-4"></a><h2>4. Mixed Integer Programming : Water Network Problem</h2>[Home](#home)

<h3>Problem definition </h3>
<br>
Consider a water network that consists of a set of nodes. Each node has a demand for water and may produce some water. The goal is to decide where to produce the water to meet the demand and to determine the best way to transport the water from the production nodes to the consumption nodes through pipelines. Each pipeline has a capacity that cannot be exceeded. There is a transportation cost for shipping a unit of water through each pipeline and a penalty for each unit of demand that is not fulfilled by the production and transportation plan. The goal is to minimize the total cost. 
    

<h3>Decision variables</h3>

- decide whether a warehouse serves a customer
     - $v_{ij}$ = amount of water flowing from node $i$ to node $j$
     - $p_i$ = amount of water produced at node $i$
     - $d_i$ = amount of water demand at node $i$
     

<h3>What are the constraints?</h3>

- Each pipeline has a capacity that cannot be exceeded \begin{align*} &v_{ij} \leq ca_{ij} & \forall i,j \\\end{align*}
- The amount of water produced by node i cannot exceed the maximum production of this node \begin{align*} &p_{i} \leq p_{max_i} & \forall i \\\end{align*}
- The amount of water produced + received by node i is equal to what is consumed (demand) + what leaves \begin{align*} &\sum_j v_{ji} + p_i = \sum_j v_{ij} + d_{i} - z_{i} & \forall i \\\end{align*}

### Problem Formulation

\begin{align*}
\min &\qquad \sum_{i,j} vc * v_{ij} + \sum_{i} p * z_{i} & \\
\end{align*}


```python
# consumption (demand)
d = [ 0, 50, 95, 10, 73, 55, 125, 32, 40, 20 ]
# production (maximum generation)
p_max = [ 500, 0, 0, 500, 0, 0, 500, 0, 0, 0 ]

N_NODES = len(d)

# capacity of the arcs
ca = [ [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
       [20, 30, 40, 50, 60, 70, 80, 90, 100, 10],
       [30, 40, 50, 60, 70, 80, 90, 100, 10, 20],
       [40, 50, 60, 70, 80, 90, 100, 10, 20, 30],
       [50, 60, 70, 80, 90, 100, 10, 20, 30, 40],
       [60, 70, 80, 90, 100, 10, 20, 30, 40, 50],
       [70, 80, 90, 100, 10, 20, 30, 40, 50, 60],
       [80, 90, 100, 10, 20, 30, 40, 50, 60, 70],
       [90, 100, 10, 20, 30, 40, 50, 60, 70, 80],
       [100, 10, 20, 30, 40, 50, 60, 70, 80, 90]
     ]

# linear variable cost: the cost of transporting one unit of water
vc = 1

# unsatisfied demand: penalty for each unit of water which is not consumed or produced
penalty = 1000
```


```python
mip_model = pywraplp.Solver.CreateSolver("SAT")


# Create variables
v = {}
for i,j in itertools.product(range(N_NODES), range(N_NODES)):
    v[i,j] = mip_model.IntVar(0, mip_model.infinity(), f"v[{i,j}]") 
    
z = {}
for i in range(N_NODES):
    z[i] = mip_model.IntVar(0, mip_model.infinity(), f"z[{i}]") 

p = {}
for i in range(N_NODES):
    p[i] = mip_model.IntVar(0, p_max[i], f"p[{i}]") 
    
# Create constraints
for i in range(N_NODES):
    mip_model.Add(sum([v[j, i] for j in range(N_NODES)]) + p[i] == sum([v[i, j] for j in range(N_NODES)]) + d[i] - z[i])
    

for i,j in itertools.product(range(N_NODES), range(N_NODES)):
    mip_model.Add(v[i, j] <= ca[i][j])
    
    
# -- Define the objective function
objective = sum([v[i, j] * vc for i, j in itertools.product(range(N_NODES), range(N_NODES))])
objective += sum([z[i] * penalty for i in range(N_NODES)])
mip_model.Minimize(objective)

print(f"Solving with {solver.SolverVersion()}")
status = mip_model.Solve()

if status == pywraplp.Solver.OPTIMAL:
    print("Objective value =", mip_model.Objective().Value())
    for i in range(N_NODES):
        if p[i].solution_value() > 0:
            print(f"Production at node {i} -> {p[i].solution_value()}")
    for i,j in itertools.product(range(N_NODES), range(N_NODES)):
        if v[i, j].solution_value() > 0:
            print(f"Flow between {i} and {j} -> {v[i, j].solution_value()}")
else:
    print("The problem does not have an optimal solution.")
```

    Solving with SCIP 9.0.0 [LP solver: Glop 9.10]
    Objective value = 365.0
    Production at node 0 -> 112.0
    Production at node 3 -> 113.0
    Production at node 6 -> 275.0
    Flow between 0 and 2 -> 5.0
    Flow between 0 and 4 -> 50.0
    Flow between 0 and 5 -> 35.0
    Flow between 0 and 7 -> 22.0
    Flow between 3 and 1 -> 50.0
    Flow between 3 and 4 -> 23.0
    Flow between 3 and 7 -> 10.0
    Flow between 3 and 8 -> 20.0
    Flow between 6 and 2 -> 90.0
    Flow between 6 and 5 -> 20.0
    Flow between 6 and 8 -> 20.0
    Flow between 6 and 9 -> 20.0



```python

```
