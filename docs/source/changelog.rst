Changelog
=========

0.4.13 / 2020-11-22
-------------------

- ``NeuronCriteria``: Added ``cellBodyFiber`` parameter.
- ``SynapseCriteria``: Changed the default value of ``primary_only`` to ``False``,
  since it may been counter-intuitive to obtain duplicate results by default.

0.4.12 / 2020-11-21
-------------------

- Better handling when adjacency queries return empty results
- Simulation: Minor change to subprocess communication implementation
- Skeleton DataFrames use economical dtypes
- Minor bug fixes and performance enhancements
- fetch_synapse_connections(): Fix pandas error in assertion

0.4.11 / 2020-06-30
-------------------

- Fixed ``ngspice`` install instructions.


0.4.10 / 2020-06-30
-------------------

- Moved skeleton-related functions into their own module, and added a few more skeleton utilty functions
- Simulation: Support Windows
- ``heal_skeleton():`` Allow caller to specify a maximum distance for repaired skeleton segments (#12)

0.4.9 / 2020-04-29
------------------

- Added simulation functions and tutorial
