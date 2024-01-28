.. _research:

Research
--------

Simulation Modelling Language
=============================

    .. code::

       >>> from Solverz import Var, as_Vars, exp, Eqn, nr_method, parse_ae_v
       >>> # Declare the unknown variables and give the initial values.
       >>> x = Var('x', 0)
       >>> # Declare your equations
       >>> E = Eqn(name='E', eqn=exp(x)+x-3)
       >>> g = AE(name='g', eqn=E)
       >>> y0 = as_Vars(x)
       >>> # Convert symbolic models to numerical functions
       >>> ng = made_numerical(g, y0)
       >>> # Solve the equation with Solverz' built-in Newton Solver
       >>> y = nr_method(ng, y0.array)
       >>> y

Digital Twin Simulation
=======================

