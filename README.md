# OOMMF_GPU
We demonstrated a GPU implementation of the widely used [Object Oriented Micromagnetic Framework (OOMMF)](http://math.nist.gov/oommf/), showing up to 35x GPU-CPU acceleration. The implementation is such that most of the user-related OOMMF components are unchanged and only the lower-level modules are ported to GPU. This allows OOMMF users to run their models as before but at greater speed.

OOMMF is a project in the Applied and Computational Mathematics Division (ACMD) of ITL/NIST, aimed at developing portable, extensible public domain programs and tools for micromagnetics.

The implementation is open-sourced. If it helps with your research, we will appreciate it if you refer the following artical in the publications.

[S. Fu, W. Cui, M. Hu, R. Chang, M.J. Donahue, V. Lomakin, "Finite Difference Micromagnetic Solvers with Object Oriented Micromagnetic framework (OOMMF) on Graphics Processing Units," Magnetics, IEEE Transactions on, vol.PP, no.99, pp.1-1](http://ieeexplore.ieee.org/xpl/abstractMetrics.jsp?arnumber=7335615&tag=1)
