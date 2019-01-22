# Introduction

**pytzer.jfuncs** provides different ways to evaluate the J and J' functions that appear in the unsymmetrical mixing terms of the Pitzer model.

One of these functions must be contained within the [cfdict](../cfdicts) in order to execute pytzer.

## jfunc syntax

Every **jfunc** must have the following basic structure:

```python
def jfunc(x):

    J  = <function of x for J>
    Jp = <derivative of J with respect to x>

    return J(x), Jp(x)
```

# jfunc options

## pytzer.jfuncs.P75_eq46



## pytzer.jfuncs.P75_eq47



## pytzer.jfuncs.Harvie
