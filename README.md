# QuaSiModO

For more *details* see the associated publication: **[S. Peitz & K. Bieker, On the Universal Transformation of Data-Driven Models to Control Systems, arXiv:2102.04722](https://arxiv.org/abs/2102.04722)**

**Authors**: Sebastian Peitz & Katharina Bieker *(Department of Mathematics, Paderborn University)*

## Example files
An **intorudctory standalone script** can be found in the *Jupyter notebook* **Standalone_Duffing_EDMD.ipynb** in the main folder.

Many different examples (including the ones from the paper) are available in the *tests* folder. The associated models can either be found in the *models* folder or - in the case of ODEs - are directly implemented in the main file.

# Short description
The QuaSiModO algorithm is designed to universally transform predictive models into discrete-time control systems with continuous inputs and a tracking-type objective function. To achieve this, we use several transformations of the optimization problem:
1. **Quantization** of the control set *U* into a finite set *V* with *m* different fixed control inputs, transforming the non-autonomous control system into *m* autonomous systems; this yields a *mixed-integer optimal control problem (MIOCP)*,
2. **Simulation** (i.e., data collection) of the *m* autonomous systems with fixed inputs,
3. **Modeling** of the *m* systems using an arbitrary surrogate modeling technique,
4. **Relaxation** of the MIOCP to a continuous control problem,
5. **Optimization** of the relaxed problem.

# Code structure
All main functionalities are implemented in *QuaSiModO.py*, and some helper functions for visualization are implemented in *visualization.py*. For more details on the different options when creating models or data sets, please refer to the introductory comments of the different class definitions.

## Models for the true system
Model files are usually created in the subfolder *models*. They need to have a function called **simulateModel(y0, t0, u, model)**, where the inputs are the initial state **y0** at initial time **t0** and the control is an array of size *(p + 1, dim(U))*, where *p* is the number of prediction steps (i.e., the time horizon). The last input **model** is of type *ClassModel* and contains additonal parameters as well as the necessary details regarding the numerical discretization etc. It needs to be created in the main file, see the examples. The function **returns y**, **z**, **t** and **model**, where **y** and **z** are the full state and the observed quantiy, respectively.

When initializing a model class by calling *ClassModel*, several variables need to be specified:
* the model input, which can be a function call (i.e., the right-hand side of an ODE) or the name of a py file stored in *models* (e.g., *'burgers.py'*)
* the time step **h** for the numerical solution of the model
* the control bounds **uMin** and **uMax**
* additional **params** (optional)
* the dimension of the observable, i.e., **dimZ**
* the type of the control quantization via **typeUGrid** (*'cube'*, *'cubeCenter'*, *'centerStar'* or *'oneSidedStar'*, see the description in *QuaSiModO.py* for details) and **nGridU** (which describes the number of discretization points in each spatial dimension)

## Surrogate Models
Surrogate models are usually created in the subfolder *surrogateModels*. They need to contain the functions 
* timeTMap
* createSurrogateModel
* updateSurrogateModel *(optional)*

**timeTMap(z0, t0, iu, modelData)** (i.e., the time-T-map of the reduced model), where **z0** and **t0** are the initial conditions, and **iu** is from *{1, ..., m}* and denotes the index of the autonomous system that is used. **modelData** is of type *ClassSurrogateModel* and contains all the necessary surrogate model information. It needs to be created beforehand. The routine **returns z**, **t** and **modelData**

**createSurrogateModel(modelData, data)** is the routine where the surrogate model is created and stored in **modelData**, using the data that is given via **data** of type *ClassControlDataSet*. Thre function **returns** and updated version of **modelData**.

**updateSurrogateModel(modelData, z, u, iu)** is used to update one or more models during the MPC routine using collected data. Here, **z**, **u** and **iu** are the time series that are used to update the model.

## Data collection
The data is stored in a variable of type *ClassControlDataSet*, e.g., **dataSet = ClassControlDataSet(h=0.1, T=10)**, where trajectories are 10 seconds long with a time increment of 0.1 seconds. 

Input trajectories are created by calling, e.g., **uTrain, iuTrain = dataSet.createControlSequence(model, typeSequence='piecewiseConstant', nhMin=1, nhMax=5)**. In this case, the input is piecewise constant, inputs remaining constant over one to five time steps. 

The data is created by simulation when calling *createData* (e.g., **dataSet.createData(model=model, u=uTrain, savePath=pathData)**, where the model is passed to routine along with the set of input trajectories and a potential path to store the data for later use).

The data to pass on to the surrogate modeling can then be prepared by calling *prepareData*, e.g., **data = dataSet.prepareData(model, method='dX', rawData=dataSet.rawData, nLag=nLag, nDelay=0)**. Here, the **method** variable says how to process the data (in this case, we get the trajectories *X* and their time derivatives *dX*; if we write 'Y', we get the trajectories and their time-shifted version. The time increment is defined by the factor **nLag** by which the time step **h** is multiplied.

# Additional tools
## OpenFOAM (Works only on Linux systems (tested with Ubuntu 18.04 and 20.04))
The fluid dynamics examples all use the open source flow solver [**OpenFOAM**](https://www.openfoam.com/). The code has been tested with the OpenFOAM version *v1912*. The required functionality is implemented in the files in the subfolder *OpenFOAM*. To use OpenFOAM, the configuration in *configOpenFOAM.py* needs to be adapted to the correct paths. The different problem setups are stored in individual folders in *OpenFOAM/problems*, and the control input has to be realized via ASCII files with names *control0*, *control1*, ..., that are located in the main folder of the problem setup.

## d3s
The toolbox [**d3s by Stefan Klus**](https://github.com/sklus/d3s) is included as well. The surrogate models using EDMD and generator EDMD make use of some of the routines implemented in d3s.
