# Python version of albedo model

This repository contains a Python version of Rautiainen et al. (2018) GAMS-model

Requirements
- Python version X
- Ipopt version X

RESULTS:
- output/ folder contains results in excel format
- Terminal output:

GAMS version results (Jussi):
Number of Iterations....: 49

                                   (scaled)                 (unscaled)
                            2.5472504543004861e+03
Objective...............:   2.5472577568742145e+03   -2.5472577568742145e+03
Dual infeasibility......:   7.6508041541305831e-08    7.6508041541305831e-08
Constraint violation....:   4.0970007830765097e-09    8.1940015661530183e-08
Variable bound violation:   9.9999506224092278e-11    9.9999506224092278e-11
Complementarity.........:   5.2755485282821695e-10    5.2755485282821695e-10
Overall NLP error.......:   4.0970007830765097e-09    8.1940015661530183e-08


Number of objective function evaluations             = 82
Number of objective gradient evaluations             = 1
Number of equality constraint evaluations            = 82
Number of inequality constraint evaluations          = 82
Number of equality constraint Jacobian evaluations   = 50
Number of inequality constraint Jacobian evaluations = 1
Number of Lagrangian Hessian evaluations             = 49
Total seconds in IPOPT                               = 7.983

EXIT: Optimal Solution Found.



Python version (Joel):
Number of Iterations....: 368

                                   (scaled)                 (unscaled)
Objective...............:   2.5472504543004861e+03    2.5472504543004861e+03
Dual infeasibility......:   7.0936908090121861e-09    7.0936908090121861e-09
Constraint violation....:   3.2102681454350166e-09    3.2102681454350166e-09
Variable bound violation:   9.9992300882454008e-09    9.9992300882454008e-09
Complementarity.........:   3.9278191073441544e-09    3.9278191073441544e-09
Overall NLP error.......:   5.9673386136405526e-09    7.0936908090121861e-09


Number of objective function evaluations             = 677
Number of objective gradient evaluations             = 324
Number of equality constraint evaluations            = 677
Number of inequality constraint evaluations          = 677
Number of equality constraint Jacobian evaluations   = 369
Number of inequality constraint Jacobian evaluations = 369
Number of Lagrangian Hessian evaluations             = 368
Total seconds in IPOPT                               = 71.233

EXIT: Optimal Solution Found.


References

Bynum, M., L., Hackebeil, G., A., Hart, W., E., Laird, C., D., Nicholson, B., L., Siirola, J., D., 
Watson, J., Woodruff, D., L. (2021). Pyomo - Optimization Modeling in Python, 3rd Edition. 
Springer.

Rautiainen A., Lintunen J., Uusivuori J. (2018). Market-level implications of regulating forest 
carbon storage and albedo for climate change mitigation. Agricultural and Resource 
Economics Review, 47(2), 239 – 271. https://doi.org/10.1017/age.2018.8

Wächter, A., Biegler, L. On the implementation of an interior-point filter line-search 
algorithm for large-scale nonlinear programming. Math. Program. 106, 25–57 (2006). 
https://doi.org/10.1007/s10107-004-0559-y
