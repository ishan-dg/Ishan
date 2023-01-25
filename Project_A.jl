using Flux, ProgressMeter
using Flux.Data: DataLoader
using Plots
using IterTools
using JLD
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
function res_loss(time,Phi)

    #(phi_train, t_train), (phi_test, phi_test) = splitobs((Phi, time); at = 0.35)
    phi_m=model(time)
    loss_train=Flux.mean((Phi .-phi_m).^2)
    time_P=(range(0,2,step=0.001))
    time_P = convert(Array{Float64}, time_P)'
    
    phi_P=model(collect(range(0,2,step=0.001))')
    r_PINN=0
        #delta_t=time_P[i+1]-time_P[i]
    
    phi_double=(phi_P[3:end] .- 2*phi_P[2:end-1] .+ phi_P[1:end-2])/((0.001)^2)
    phi_dot=(phi_P[3:end].-phi_P[1:end-2])/(2*0.001)
    r_PINN=(phi_double.+5*phi_dot.+300*phi_P[2:end-1])
    return loss_train+1e-6*Flux.mean(abs2.(r_PINN))
end

trainer= DataLoader((time, Phi), shuffle=true, batchsize=40)#(t_phy,del_t)
training_loss = Float64[]
testing_loss = Float64[]
epochs = Int64[]
iterations=500
for epoch in ProgressBar(1:iterations)
    Flux.train!(res_loss, Flux.params(model), trainer, optimizer)   # Training call
    if epoch % 10 == 0
        push!(epochs, epoch)
    end
end
plot(time',[Phi' model(time)'])
if save_plot == true
    println("Saving Data Plot")
    plot(time',[Phi' model(time)'])
    savefig(case * ".png")
    println("Save Complete")
end
