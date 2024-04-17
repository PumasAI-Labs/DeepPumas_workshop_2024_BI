using DeepPumas
using StableRNGs
using CairoMakie
using Serialization
using Latexify
using PumasPlots
set_theme!(deep_light())

############################################################################################
## Generate synthetic data from common pharmacokinetic model with nonlinear drug clearance
############################################################################################

datamodel = @model begin
  @param begin
    tvImax ∈ RealDomain(; lower=0.)
    tvIC50 ∈ RealDomain(; lower=0.)
    tvKa ∈ RealDomain(; lower=0.)
    tvVc ∈ RealDomain(; lower=0.)
    Ω ∈ PDiagDomain(4)
    σ ∈ RealDomain(; lower=0.)
  end
  @random η ~ MvNormal(Ω)
  @covariates AGE Weight c
  @pre begin
    Ka = tvKa * exp(η[1]) +  0.5 * (AGE/55)^2 
    Imax = tvImax * exp(η[2]) + 1.6*(Weight + AGE)/Weight 
    IC50 = tvIC50 * exp((Weight/75)^2 + η[3])
    Vc = tvVc * exp(η[4] + Weight/75 * c / (1 + c)) 
  end  
  @dynamics begin
    Depot' = - Ka * Depot
    Central' = Ka * Depot - Imax * (Central/Vc) / (IC50 + (Central/Vc))
  end
  @derived begin
    Concentration ~ @. Normal(Central/Vc, σ)
  end
end

# The covariate effect on the individual parameters are a bit crazy on purpose
latexify(datamodel, :pre) |> render

p_data = (;
  tvImax = 1.1,
  tvIC50 = 0.4,
  tvKa = 1.,
  tvVc = 0.8,
  Ω = Diagonal([0.1, 0.1, 0.1, 0.1]),
  σ = 0.1,
)

rng = StableRNG(1)
pop = map(1:1012) do i
  _subj = Subject(;
    id = i,
    covariates=(;
      AGE=rand(rng, truncated(Normal(55,10), 15, Inf)),
      Weight=rand(rng, truncated(Normal(75,25), 20, Inf)),
      c = rand(rng, Gamma(4, 0.1))
    ),
    events = DosageRegimen(8., ii=2, addl=1)
  )
  sim = simobs(datamodel, _subj, p_data; obstimes = 0:0.5:8)
  Subject(sim)
end

## Split the data into different training/test populations

trainpop_small = pop[1:50]
trainpop_large = pop[1:1000]
testpop = pop[1001:end]

pred_datamodel = predict(datamodel, testpop, p_data; obstimes=0:0.01:8);
plotgrid(pred_datamodel)


############################################################################################
## Neural-embedded NLME modeling
############################################################################################
# Here, we define a model where the PD is entirely deterimined by a neural network.
# At this point, we're not trying to explain how patient data may inform individual
# parameters



model = @model begin
  @param begin
    tvKa ∈ RealDomain(; lower=0.)
    
    # A multi-layer perceptron (MLP) with
    # - 4 inputs
    # - one hidden layer of 4 nodes and tanh activation (default)
    # - another hidden layer with 4 nodes and tanh activation
    # - one output node with identity activation
    # And L2 regularization
    NN ∈ MLPDomain(4, 4, 4, (1, identity); reg=L2(1.))
    ω_ka ∈ RealDomain(lower=0.)
    Ω ∈ PDiagDomain(2)
    σ ∈ RealDomain(; lower=0., init=0.08)
  end
  @random begin
    η_ka ~ Normal(0., ω_ka)
    η ~ MvNormal(Ω)
  end
  @pre begin
    Ka = tvKa * exp(η_ka)
    iNN = fix(NN, η)
  end  
  @dynamics begin
    Depot' = - Ka * Depot
    Central' = Ka * Depot - iNN(Central, Depot)[1]
  end
  @derived begin
    Concentration ~ @. Normal(Central, σ)
  end
end


fpm = fit(
  model,
  trainpop_small,
  init_params(model),
  MAP(FOCE());
  # Some extra options to speed up the demo at the expense of a little accuracy:
  optim_options=(; iterations=300),
  constantcoef = (; Ω = Diagonal(fill(0.1, 2)))
)

# The model has succeeded in discovering the dynamical model if the individual predictions
# match the observations of the test population well.
pred = predict(model, testpop, coef(fpm); obstimes=0:0.01:8);
plotgrid(pred)

############################################################################################
## 'Augment' the model to predict heterogeneity from data
############################################################################################
# All patient heterogeneity of our recent model was captured by random effects and can thus
# not be predicted by the model. Here, we 'augment' that model with ML that's trained to 
# capture this heterogeneity from data.

# Generate a target for the ML fitting from a Normal approximation of the posterior η
# distribution.
target = preprocess(fpm)

nn = MLPDomain(numinputs(target), 7, 7, (numoutputs(target), identity); reg=L2(1.0))

fnn = fit(nn, target)
augmented_fpm = augment(fpm, fnn)

pred_augment =
  predict(augmented_fpm.model, testpop, coef(augmented_fpm); obstimes=0:0.01:8);
plotgrid(
  pred_datamodel;
  ipred=false,
  pred=(; color=(:black, 0.4), label="Best possible pred")
)
plotgrid!(pred; ipred=false, pred=(; color=(:red, 0.2), label="No covariate pred"))
plotgrid!(pred_augment; ipred=false, pred=(; linestyle=:dash))

# Define a function to compare pred values so that we can see how close our preds were to
# the preds of the datamodel
function pred_residuals(pred1, pred2)
  mapreduce(hcat, pred1, pred2) do p1, p2
    p1.pred.Concentration .- p2.pred.Concentration
  end
end


residuals = pred_residuals(pred_datamodel, pred_augment)
mean(abs, residuals)

# residuals between the preds of no covariate model and the preds of the datamodel 
residuals_base = pred_residuals(pred_datamodel, pred)
mean(abs, residuals_base)

# We should now have gotten some improvement over not using covariates at all. However,
# training covariate models well requires more data than fitting the neural networks
# embedded in dynamical systems. With UDEs, every observation is a data point. With
# prognostic factor models, every subject is a data point. We've (hopefully) managed to
# improve our model using only 50 subjects, but lets try using data from 1000 patients
# instead. 

target_large = preprocess(model, trainpop_large, coef(fpm), FOCE())
fnn_large = hyperopt(nn, target_large)
augmented_fpm_large = augment(fpm, fnn_large)


pred_augment_large = predict(augmented_fpm_large, testpop; obstimes=0:0.01:8);
plotgrid(
  pred_datamodel;
  ipred=false,
  pred=(; color=(:black, 0.4), label="Best possible pred")
)
plotgrid!(pred; ipred=false, pred=(; color=(:red, 0.2), label="No covariate pred"))
plotgrid!(pred_augment_large; ipred=false, pred=(; linestyle=:dash))

# residuals between the preds of no covariate model and the preds of the datamodel 
residuals_large = pred_residuals(pred_datamodel, pred_augment_large)
mean(abs, residuals_large)


# If we've really uncovered something like the true model, then we should be able to predict
# what would happen in scenarios we've never trained for, right?

rng = StableRNG(2)
pop_new_dose = map(1:12) do i
  _subj = Subject(;
    id = i,
    covariates=(;
      AGE=rand(rng, truncated(Normal(55,10), 15, Inf)),
      Weight=rand(rng, truncated(Normal(75,25), 20, Inf)),
      c = rand(rng, Gamma(4, 0.1))
    ),
    events = DosageRegimen(4., ii=1, addl=4)
  )
  sim = simobs(datamodel, _subj, p_data; obstimes = 0:0.5:8)
  Subject(sim)
end
plotgrid(pop_new_dose)

pred_new_dose = predict(augmented_fpm_large, pop_new_dose; obstimes=0:0.01:8)
pred_datamodel_new_dose = predict(datamodel, pop_new_dose, p_data; obstimes=0:0.01:8)

plotgrid(
  pred_datamodel_new_dose;
  ipred=false,
  pred=(; color=(:black, 0.4), label="Best possible pred")
)
plotgrid!(pred_new_dose; ipred=false)