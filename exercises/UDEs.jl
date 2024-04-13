using DeepPumas
using CairoMakie
using StableRNGs
set_theme!(deep_light())
set_theme!(deep_dark())

# 
# TABLE OF CONTENTS
# 
# 1. INTRODUCTION
#
# 1.1. Simulate subjects A and B with different dosage regimens
# 1.2. A dummy neural network for modeling dynamics
# 
# 2. IDENTIFICATION OF MODEL DYNAMICS USING NEURAL NETWORKS
#
# 2.1. Identify the dynamics with a NeuralODE
# 2.2. Identify only the PD using a universal differential equation (UDE)
# 2.4. Extend the analysis to a population of multiple subjects
# 2.5. Use a UDE model on sparse timecourses from multiple patients
# 2.6. Use a UDE model on rich timecourses from multiple patients
#

# 
# 1. INTRODUCTION
#
# 1.1. Simulate subjects A and B with different dosage regimens
# 1.2. A dummy neural network for modeling dynamics
# 

"""
Helper Pumas model to generate synthetic data. It assumes 
one compartment non-linear elimination and oral dosing.
"""
data_model = @model begin
  @param begin
    tvKa ∈ RealDomain()
    tvCL ∈ RealDomain()
    tvVc ∈ RealDomain()
    tvSmax ∈ RealDomain()
    tvn ∈ RealDomain()
    tvSC50 ∈ RealDomain()
    tvKout ∈ RealDomain()
    tvKin ∈ RealDomain()
    σ ∈ RealDomain()
  end
  @pre begin
    Smax = tvSmax
    SC50 = tvSC50
    Ka = tvKa
    Vc = tvVc
    Kout = tvKout
    Kin = tvKin
    CL = tvCL
    n = tvn
  end
  @init begin
    R = Kin / Kout
  end
  @vars begin
    cp = max(Central / Vc, 0.0)
    EFF = Smax * cp^n / (SC50^n + cp^n)
  end
  @dynamics begin
    Depot' = -Ka * Depot
    Central' = Ka * Depot - (CL / Vc) * Central
    R' = Kin * (1 + EFF) - Kout * R
  end
  @derived begin
    Outcome ~ @. Normal(R, abs(R) * σ)
  end
end

true_parameters = (;
  tvKa=0.5,
  tvCL=1.0,
  tvVc=1.0,
  tvSmax=2.9,
  tvn=1.5,
  tvSC50=0.05,
  tvKout=2.2,
  tvKin=0.8,
  σ=0.02                         ## <-- tune the observational noise of the data here
)

# 1.1. Simulate subjects A and B with different dosage regimens
data_a = begin
  _subj = Subject(; events=DosageRegimen(1.0, addl=1, ii=5), id="Subject A")
  sim = simobs(data_model, _subj, true_parameters; obstimes=0:0.5:15, rng=StableRNG(1))
  [Subject(sim)]
end

data_b = begin
  _subj = Subject(; events=DosageRegimen(0.2, addl=1, ii=5), id="Subject B")
  sim = simobs(data_model, _subj, true_parameters; obstimes=0:0.5:15, rng=StableRNG(2))
  [Subject(sim)]
end


plotgrid(data_a; data=(; label="Data (subject A)"), title=false)
plotgrid!(data_b; data=(; label="Data (subject B)"), color=:gray, title=false)

pred_datamodel_a = predict(data_model, data_a, true_parameters; obstimes=0:0.01:15)
pred_datamodel_b = predict(data_model, data_b, true_parameters; obstimes=0:0.01:15)
plotgrid(pred_datamodel_a; ipred=false)
plotgrid!(pred_datamodel_b; data=true, ipred=false)


# 1.2. A non-dynamic machine learning model for later comparison.

```
    time_model
    
A machine learning model mapping time to a noisy outcome. This is not a SciML model.
```
time_model = @model begin
  @param begin
    mlp ∈ MLPDomain(1, 6, 6, (1, identity); reg=L1(1.0; output=false))
    σ ∈ RealDomain(; lower=0.0)
  end
  @derived begin
    nn_output := first.(mlp.(t))
    # unpacking that call:
    # t       - a vector of all time points for which a subject had observations. 
    # mlp.(t) - apply the mlp on each element of t. (the . "broadcasts" the function over all elements instead of using the vector directly) 
    # first   - get the first element of the mlp output (the output is a 1-element vector)
    Outcome ~ @. Normal(nn_output, σ)
  end
end

# Strip the dose out of the subject since this simple model does not know what to do with a dose.
data_a_no_dose = Subject.(data_a; events=nothing)
data_b_no_dose = Subject.(data_b; events=nothing)

fpm_time = fit(time_model, data_a_no_dose, init_params(time_model), MAP(NaivePooled()))

pred_a = predict(fpm_time; obstimes=0:0.1:15);
plotgrid(
  pred_a;
  pred=(; label="Pred (subject A)"),
  data=(; label="Data (subject A)"),
  ipred=false
)

pred_b = predict(time_model, data_b_no_dose, coef(fpm_time); obstimes=0:0.1:15);
plotgrid!(
  pred_b,
  pred=(; label="Pred (subject B)", color=:red, linestyle=:dash),
  data=(; label="Data (subject B)"),
  ipred=false,
)

# 
# 2. IDENTIFICATION OF MODEL DYNAMICS USING NEURAL NETWORKS
#
# 2.1. Identify the dynamics with a NeuralODE
# 2.2. Identify only the PD using a universal differential equation (UDE)
# 2.3. Improve the UDE model by encoding more domain knowledge
# 2.4. Extend the analysis to a population of multiple subjects
# 2.5. Use a UDE model on sparse timecourses from multiple patients
# 2.6. Use a UDE model on rich timecourses from multiple patients
#

# 2.1. Delegate the identification of dynamics to a neural network

preds_a = Dict()
preds_b = Dict()

neural_ode_model = @model begin
  @param begin
    mlp ∈ MLPDomain(3, 6, 6, (3, identity); reg=L1(1.0; output=false))    # neural network with 2 inputs and 1 output
    tvR₀ ∈ RealDomain(; lower=0)
    σ ∈ RealDomain(; lower=0)                       # residual error
  end
  @pre begin
    mlp_ = mlp
    R₀ = tvR₀
  end
  @init R = R₀
  @dynamics begin
    Depot' = mlp_(Depot, Central, R)[1]
    Central' = mlp_(Depot, Central, R)[2]
    R' = mlp_(Depot, Central, R)[3]
  end
  @derived begin
    Outcome ~ @. Normal(R, abs(R) * σ)
  end
end

fpm_node = fit(neural_ode_model, data_a, init_params(neural_ode_model), MAP(NaivePooled()))

preds_a[:NODE] = predict(fpm_node; obstimes=0:0.01:15)
plotgrid(
  preds_a[:NODE];
  pred=(; label="Pred (subject A)"),
  ipred=false,
  data=(; label="Data (subject A)"),
)

preds_b[:NODE] = predict(fpm_node, data_b; obstimes=0:0.01:15)
plotgrid!(
  preds_b[:NODE],
  pred=(; label="Pred (subject B)", color=:red),
  data=(; label="Data (subject B)"),
  ipred=false,
)

# You can get pretty good results here but the generalization performance is rather brittle.
# Try changing the the parameters from init_params (deterministic) to sample_params (random
# and anything goes) and fit again a few times. How well do you fit subject A? And how well
# do you fit subject B?
# What about changing the number of hidden nodes in the neural network?


# 2.2. Identify only the PD using a universal differential equation (UDE)
#
# Let's encode some more knowledge, leaving less for the neural network to pick up.

ude_model = @model begin
  @param begin
    mlp ∈ MLPDomain(2, 6, 6, (1, identity); reg=L1(1.0))    # neural network with 2 inputs and 1 output
    tvKa ∈ RealDomain(; lower=0)                    # typical value of absorption rate constant
    tvCL ∈ RealDomain(; lower=0)
    tvVc ∈ RealDomain(; lower=0)
    tvR₀ ∈ RealDomain(; lower=0)
    σ ∈ RealDomain(; lower=0)                       # residual error
  end
  @pre begin
    mlp_ = only ∘ mlp # equivalent to (args...) -> only(mlp(args...))
    CL = tvCL
    Vc = tvVc
    Ka = tvKa
    R₀ = tvR₀
  end
  @init R = R₀
  @dynamics begin
    Depot' = -Ka * Depot                                # known
    Central' = Ka * Depot - (CL / Vc) * Central
    R' = mlp_(Central / Vc, R)
  end
  @derived begin
    Outcome ~ @. Normal(R, abs(R) * σ)
  end
end

fpm_ude = fit(ude_model, data_a, init_params(ude_model), MAP(NaivePooled()))

preds_a[:UDE] = predict(fpm_ude; obstimes=0:0.1:15);
plotgrid(
  preds_a[:UDE];
  pred=(; label="Pred (subject A)"),
  data=(; label="Data (subject A)"),
  ipred=false,
)

preds_b[:UDE] = predict(ude_model, data_b, coef(fpm_ude); obstimes=0:0.1:15);
plotgrid!(
  preds_b[:UDE],
  pred=(; label="Pred (subject B)", color=:red),
  data=(; label="Data (subject B)"),
  ipred=false,
)


plotgrid!(pred_datamodel_a; pred=(; color=(:black, 0.4), label="Datamodel"), ipred=false)
plotgrid!(pred_datamodel_b; pred=(; color=(:black, 0.4), label="Datamodel"), ipred=false)



# 2.4. Extend the analysis to a population of multiple subjects

data_model_heterogeneous = @model begin
  @param begin
    tvKa ∈ RealDomain()
    tvCL ∈ RealDomain()
    tvVc ∈ RealDomain()
    tvSmax ∈ RealDomain()
    tvn ∈ RealDomain()
    tvSC50 ∈ RealDomain()
    tvKout ∈ RealDomain()
    tvKin ∈ RealDomain()
    σ ∈ RealDomain()
  end
  @random begin
    η ~ MvNormal(5, 0.2)
  end
  @pre begin
    Smax = tvSmax * exp(η[1])
    SC50 = tvSC50 * exp(η[2])
    Ka = tvKa * exp(η[3])
    Vc = tvVc * exp(η[4])
    Kout = tvKout * exp(η[5])
    Kin = tvKin
    CL = tvCL
    n = tvn
  end
  @init begin
    R = Kin / Kout
  end
  @vars begin
    cp = max(Central / Vc, 0.0)
    EFF = Smax * cp^n / (SC50^n + cp^n)
  end
  @dynamics begin
    Depot' = -Ka * Depot
    Central' = Ka * Depot - (CL / Vc) * Central
    R' = Kin * (1 + EFF) - Kout * R
  end
  @derived begin
    Outcome ~ @. Normal(R, σ)
  end
end


# 2.5. Use a UDE model on sparse timecourses from multiple patients

sims_sparse = [
  simobs(
    data_model_heterogeneous,
    Subject(; events=DosageRegimen(1.0), id=i),
    true_parameters;
    obstimes=11 .* sort!(rand(StableRNG(i), 2))
  ) for i = 1:30
]
population_sparse = Subject.(sims_sparse)
plotgrid(population_sparse)

fpm_sparse = fit(
  ude_model,
  population_sparse,
  init_params(ude_model),
  MAP(NaivePooled()),
)

pred = predict(fpm_sparse; obstimes=0:0.01:15);
plotgrid(pred)

# plot them all stacked ontop of oneanother
fig = Figure();
ax = Axis(fig[1, 1]; xlabel="Time", ylabel="Outcome", title="Stacked predictions")
for i in eachindex(pred)
  plotgrid!([ax], pred[i:i]; data=(; color=Cycled(i)))
end
fig

# Does it look like we've found anything reasonable?


# 2.6. Use a UDE model on rich timecourses from multiple patients
population = begin
  _subj = [Subject(; events=DosageRegimen(1.0), id=i) for i in 1:25]
  sim = simobs(data_model_heterogeneous, _subj, true_parameters; obstimes=0:1:10, rng=StableRNG(3))
  Subject.(sim)
end


fpm_ude_2 = fit(
  ude_model,
  population,
  init_params(ude_model),
  MAP(NaivePooled()),
)

pred = predict(fpm_ude_2; obstimes=0:0.1:10);
plotgrid(pred)




############################################################################################
#                                    3. Bonus material                                     #
############################################################################################

# The examples above illustrate the core concepts that we want to teach. However, they're a
# bit cleaner than one might expect in real life and they avoid some of the issues that a
# modeler may face when using UDEs/NeuralODEs.
# Here, we go through some of the problems one is likely to face when using UDEs in real
# projects and how to think when trying to solve them.

# 3.1. Akward scales 

# Most neural networks work best if the input and target outputs have value that are not too
# far from the relevant bits of our activation functions. A farily standard practice in ML
# regression is to standardize input and output to have a mean 0 and std=1 or to ensure that
# all values are between 0 and 1. With bad input/output scales, it can be hard to fit a
# model. 

scaling = 1e4
parameters_scaled = (;
  tvKa=0.5,
  tvCL=1.0,
  tvVc=1.0,
  tvSmax=1.9,
  tvn=1.5,
  tvSC50=0.05 * scaling,
  tvKout=2.2,
  tvKin=0.8 * scaling,
  σ=0.02
)

data_hard_scale = begin
  _subj = Subject(; events=DosageRegimen(scaling, addl=1, ii=5))
  sim = simobs(data_model, _subj, parameters_scaled; obstimes=0:0.5:15, rng=StableRNG(1))
  [Subject(sim)]
end

plotgrid(data_hard_scale)

fpm_hard_scale = fit(
  ude_model, # note that we're now using the model with R' = mlp(Central/Vc, R)
  data_hard_scale,
  # init_params(ude_model_knowledge),
  sample_params(ude_model),
  MAP(NaivePooled())
)

pred_hard_scale = predict(fpm_hard_scale; obstimes=0:0.1:10)
plotgrid(pred_hard_scale)

## Why did that fail so miserably? 
# We just applied a dose of 10,000 which will make the value of Central large. Furthermore,
# "Outcome" needs to be on the scale of 10,000 to fit the data and this is also an input to
# the neural network.

# But our activation function, tanh, saturates for values much larger than 1.
x = -5:0.1:5
lines(x, tanh.(x))

using DeepPumas.ForwardDiff: derivative
derivative(tanh, 0.0)
derivative(tanh, 1.0)
derivative(tanh, 10.0)
derivative(tanh, 100.0) # the gradient vanishes at large input.

lines(x, map(_x -> derivative(tanh, _x), x); axis=(; ylabel="Derivative", xlabel="input"))

## So, what'll happen in the first layer of the neural network?
w = rand(1, 6)
b = rand(6)
input = [1.0]
tanh.(w' * input .+ b)

input_large = 1e4
tanh.(w' * input_large .+ b) # All saturated, almost no gradient, no chance for the optimiser to work.

## So, what's the solution? Abandon tanh?

softplus(1e4)
derivative(softplus, 1e3)

# Looks fine? Here, we don't saturate and we have a non-zero gradient.
# We can try:

model_softplus = @model begin
  @param begin
    mlp ∈ MLPDomain(2, 6, 6, (1, identity, false); reg=L1(1), act=softplus)
    tvCL ∈ RealDomain(; lower=0)
    tvVc ∈ RealDomain(; lower=0)
    tvKa ∈ RealDomain(; lower=0)
    tvR₀ ∈ RealDomain(; lower=0, init=1e3)
    σ ∈ RealDomain(; lower=0)
  end
  @pre begin
    mlp_ = only ∘ mlp
    CL = tvCL
    Vc = tvVc
    Ka = tvKa
    R₀ = tvR₀
  end
  @init R = R₀
  @dynamics begin
    Depot' = -Ka * Depot
    Central' = Ka * Depot - (CL / Vc) * Central
    R' = mlp_(Central / Vc, R)
  end
  @derived begin
    Outcome ~ @. Normal(R, σ)
  end
end

fpm_softplus = fit(
  model_softplus,
  data_hard_scale,
  init_params(model_softplus),
  MAP(NaivePooled());
)

plotgrid(predict(fpm_softplus; obstimes=0:0.1:10))

# Hmm, the gradients are better but the model still has a hard time finding a good solution. 

# If you found something that looks pretty reasonable then odds are that you've still just found a linear relationship between the drug and the response.

nn = only ∘ coef(fpm_softplus).mlp
x = 0:10:10000
lines(x, nn.(x, 1e3); axis=(; ylabel="mlp output", xlabel="Central/Vc"))

# Even if softplus does not saturate, the magnitude of the input is still so large that it is essentially piecewise linear
lines(-2:0.1:2, softplus.(-2:0.1:2))
lines(-10000:100:10000, softplus.(-10000:100:10000))
# A piecewise linear activation function like this (relu) works well for large neural networks but poorly for small. 
# One could imagine that the neural network fitting would figure out that it ought to use
# the input layer to scale the inputs down such that the inputs to the hidden layers are ok
# and then to scale the ouputs of the output layer up again. But this is hard for two
# reasons. First, the gradient in that "direction" of parameter space is almost nonexistent.
# Softplus with an input of 1e4 is about as linear as softplus with input of 1e3. Second,
# we're regularizing the parameters to have low values. Scaling up output from ≈1 to ≈10,000
# would come with a massive penalty from our regularization. You can try changing L1(1.) to
# L1(1.; input=false, output=false) to prevent regularization of the input and output
# layers. That would remove the penalty for the NN automatically rescaling the input but
# you'd still be left with a terrible gradient for the optimizer to work with.


# With UDEs/NeuralODEs, we don't always know exactly what input values the NN will recieve,
# but we can often figure out which order of magnitude they'll have. If we can rescale the
# NN inputs and outputs to be close to 1 then we would be in a much better place. In this
# case, we know that we're dosing with 1e4 and that there's concervation from Depot to
# Central. 


model_rescale = @model begin
  @param begin
    mlp ∈ MLPDomain(2, 6, 6, (1, identity); reg=L1(1))
    tvCL ∈ RealDomain(; lower=0)
    tvVc ∈ RealDomain(; lower=0)
    tvKa ∈ RealDomain(; lower=0)
    tvR₀ ∈ RealDomain(; lower=0, init=1e3)
    σ ∈ RealDomain(; lower=0)
  end
  @pre begin
    mlp_ = only ∘ mlp
    CL = tvCL
    Vc = tvVc
    Ka = tvKa
    R₀ = tvR₀
  end
  @init R = R₀
  @dynamics begin
    Depot' = -Ka * Depot
    Central' = Ka * Depot - (CL / Vc) * Central
    R' = mlp_(Central / (Vc * 1e4), R / 1e4) * 1e4
  end
  @derived begin
    Outcome ~ @. Normal(R, abs(R) * σ)
  end
end


fpm_rescale = fit(
  model_rescale,
  data_hard_scale,
  sample_params(model_rescale), # random - you'll get different solutions every time
  MAP(NaivePooled()),
)

plotgrid(predict(fpm_rescale; obstimes=0:0.1:10))



# So, be mindful of what scales you expect your nerual network to get as inputs and to need
#to get as outputs. Also, be mindful of how the regularization may be penalizing automatic
#rescaling of the input/output layer. Here, we looked at large inputs which could have been
#solved by the weights of the first neural network being small but where the later need to
#up-scale in the output layer would be penalized by the regularization. For inputs much
#smaller than 1, we get that the necessary large weights of the input layer may be
#over-regularized. It often makes sense not to regularize the input or output layer of the
#neural network. That avoids this particular problem but it does not always make it easy to
#find the solution since initial gradients may be close to zero and the optimizer won't know
#what to do.