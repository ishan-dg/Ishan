using Flux, ProgressMeter
using Flux.Data: DataLoader
using Plots
using IterTools
using JLD
using Statistics
using Random
using MLDataUtils
using LinearAlgebra
using Tracker
using RDatasets
using Flux: params
using IterTools
import Random
using ProgressBars
save_plot = false

k=300
c=5
m=1
g=9.8
batch_size = 64
include("Integrator.jl")

int=Integrator
include("Dynsys.jl")
pendulum = Dynsys.Math_pendulum(1.0, 9.81, 1.0, 300.0, 5.0, 1, 0.0)     #l, g, m, k, c, phi, phi_dot  
Random.seed!(64)
time=load("times.jld")["t"]'
time = convert(Array{Float64}, time)
Phi = load("Data.jld")["Data"]'
Phi = convert(Array{Float64}, Phi)
Phi_train = Phi[1:floor(Int, 0.35 * length(Phi))]
Phi_train = reshape(Phi_train, 1, :)
Phi_test = Phi[1:floor(Int, 0.35 * length(Phi))]
Phi_test = reshape(Phi_train, 1, :)
time_train = time[1:floor(Int, 0.35 * length(time))]
time_train = reshape(time_train, 1, :)
time_test = time[1:floor(Int, 0.35 * length(time))]
time_test = reshape(time_train, 1, :)
    

hidden = 40
model = Chain(
    Dense(1, hidden, Flux.tanh),  
    Dense(hidden, hidden, Flux.tanh),
    Dense(hidden, hidden, Flux.tanh),
    Dense(hidden, hidden, Flux.tanh),
    Dense(hidden, 1, Flux.tanh)
)
optimizer = ADAM()
time_P=[]
PINN=[]
function res_loss(time,Phi)

    
    # phi_train = Phi[1:floor(Int, 0.35 * length(Phi))]
    # phi_train = reshape(Phi_train, 1, :)
    # time_train = time[1:floor(Int, 0.35 * length(time))]
    # time_train = reshape(time_train, 1, :)
    phi_m=model(time)
    loss_train=mean(time.-phi_m).^2
    time_P=(range(0,2,step=0.001))
    time_P = convert(Array{Float64}, time_P)'
    delta_t=time_P[2]-time_P[1]
    phi_P=model(collect(range(0,2,step=0.001))')
    r_PINN=0
        #delta_t=time_P[i+1]-time_P[i]
    for i in 3:2000
    phi_double=(phi_P[3:i] .- 2*phi_P[2:i-1] .+ phi_P[1:i-2])/((delta_t)^2)
    phi_dot=(phi_P[3:i].-phi_P[1:i-2])/(2*delta_t)
    r_PINN=((phi_double.+pendulum.c*phi_dot.+pendulum.k*phi_P[2:i-1]).^2)/i
    return loss_train+1e-4*mean((r_PINN))
    print(r_PINN)
    end
end

trainer= DataLoader((time_train, Phi_train), shuffle=true, batchsize=40)#(t_phy,del_t)
training_loss = Float64[]
testing_loss = Float64[]
epochs = Int64[]
epochs=100
for epochs in ProgressBar(1:epochs)
    Flux.train!(res_loss, Flux.params(model), trainer, optimizer)   # Training call
end
plot(time',[Phi' model(time)'],label="Training vs PINN")
if save_plot == true
    println("Saving Data Plot")
    plot(time',[Phi' model(time)'])
    savefig(case * ".png")
    println("Save Complete")
end