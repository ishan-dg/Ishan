
using PrettyTables
using Plots, Printf
using DelimitedFiles
using Pkg
#Pkg.add("JLD")
using JLD

show_video = true
println("-- pendulum euler/central diff --")
include("Dynsys.jl")
pendulum = Dynsys.Math_pendulum(1.0, 9.81, 1.0, 300.0, 5.0, 1, 0.0)     #l, g, m, k, c, phi, phi_dot      # Hier wird ein Objekt Namens pendulum der Klasse Math_pendelum erzeugt 
Integ = Dynsys.Integrator(0.001, 2000)       #sieht gut aus     
Integ.res_phi = zeros(Integ.timesteps)
Integ.res_phi_dot = zeros(Integ.timesteps)

fig = Dynsys.create_fig(pendulum)
Dynsys.plot_state(pendulum)
display(fig)
Phi = []
times = []
for i in 1:Integ.timesteps
    fig = Dynsys.create_fig(pendulum)
    x = Dynsys.run_step(Integ, "central_diff", pendulum)
    Dynsys.plot_state(pendulum)
    display(fig)
    Integ.res_phi[i] = pendulum.phi
    Integ.res_phi_dot[i] = pendulum.phi_dot
    println(Integ.res_phi[i])
    append!(Phi, Integ.res_phi[i])            
    append!(times, i * Integ.delta_t)      
end
phaseShift = []

save("Data.jld", "Data", Phi)
save("times.jld", "t", times)

for i in 1:Integ.timesteps      #This is putting the phaseshift value into an vector, so that we can add or subtract the valcue from the plotmatrix
    append!(phaseShift, pi / 2)
end

# plot(time, [a, 0.5 * sin.((time) * (sqrt(pendulum.k)) + phaseShift)])

plot(times,Phi)
