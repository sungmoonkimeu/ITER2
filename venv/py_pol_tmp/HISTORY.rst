=======
History
=======

0.1.1 (2018-12-22)
------------------

* First release on PyPI in pre-alpha state.


0.1.3 (2019-01-22)
------------------

alpha state

* Jones_vector, Jones_matrix, Stokes works.
* Mueller is in progress.
* Functions = 9/10
* Documentation = 8/10
* Tutorial = 7/10.
* Examples = 6/10.
* Drawing = 0/10.


0.1.4 (2019-02-03)
------------------

alpha state

* Jones_vector, Jones_matrix, Stokes works.
* Mueller is in progress.
* Functions = 9/10
* Documentation = 8/10
* Tutorial = 8/10.
* Examples = 8/10.
* Tests = 8/10
* Drawing = 10/10. Finished. Polarization ellipse for Jones and Stokes (partially random). Stokes on Poincaré sphere.


0.1.5 (2019-02-25)
------------------

alpha state

* Jones_vector, Jones_matrix, Stokes works.
* Jones_vector: simplify function to represent better Jones vectors.
* tests drawing: Made tests for drawing

* Mueller is in progress.
* Functions = 9/10
* Documentation = 8/10
* Tutorial = 8/10.
* Examples = 8/10.
* Tests = 8/10
* Drawing = 10/10. Finished. Polarization ellipse for Jones and Stokes (partially random). Stokes on Poincaré sphere.


0.2.0 (2019-05-25)
------------------

beta state

* Upgrade to Python 3
* Stable version including tests


0.2.1 (2019-09-04)
------------------

beta state

* Bug fixes.
* Solve incidents.
* Start to homogenize structures for both Jones and Stokes.


0.2.2 (2019-09-04)
------------------

beta state

* Bug fixes.

1.0.0 (2020-06-04)
-------------------
py_pol multidimensional alpha states

This is a big overhaul with many changes. All of them are based on the possibility of storing several vector/matrices in the same object. This reduces significantly the time required to perform the same operation to multiple vectors/matrices, using numpy methods instead of for loops. We have calculated that the reduction is around one order of magnitude.

New methods have been introduced. First, methods available for Mueller / Stoes modules have been created also for Jones (when possible). Also, some bugs and errors in the calculations have been solved.

Finally, some method and argument names have been changed to be consistent between different classes. Also, the default value of arguments with the same name have also been unified.

The biggest TO DO we have are tests. Right now, we only have tests for the Jones_vector class. However, we thought that it would be useful to release this version so the community can use it.

NOTE: Due to the change of argument and method names, this version is not compatible with the previous ones.

1.0.2 (2020-07-04)
--------------------
* Implemented workaround of axis_equal issue.

1.0.3 (2020-11-16)
--------------------
* Fixed axis_equal issue.
