####################################
# Explicit Euler
#
# numeric integration file for the
# mathematical pendulum
#
# - explicit euler
# -
####################################


mutable struct Integrator
    delta_t::Float64
    timesteps::Int64
    Integrator(delta_t, timesteps) = new(delta_t, timesteps)
    res_phi::Vector
    res_phi_dot::Vector
end

## run one integration time step
function run_step(int::Integrator, type, pendulum)          # int::Integrator Creates an Object Integrator with the Name int
    if type == "euler"
        run_euler_step(int, pendulum)
    elseif type == "central_diff"
        #run_central_diff_step(mp, pendulum)   #Original Code
        run_central_diff_step(int, pendulum)
    else
        println("... integration type not understood ...")
    end
end

## euler integration time step (homework)
function run_euler_step(int::Integrator, pendulum)
    println("Running euler step")

    ###### (homework) ######
   # pendulum.phi = pendulum.phi + pendulum.phi_dot * int.delta_t
   # pendulum.phi_dot = pendulum.phi_dot - int.delta_t * pendulum.g / pendulum.l
     pre_phi_dot_dot = -sqrt(pendulum.k)*pendulum.phi-pendulum.c * pendulum.phi_dot
     pendulum.phi_dot = pendulum.phi_dot + int.delta_t  * pre_phi_dot_dot
     pendulum.phi = pendulum.phi + int.delta_t * pendulum.phi_dot


end

## central difference time step (homework)
function run_central_diff_step(int::Integrator, pendulum)
    println("Running central difference step")
    ###### (homework) ######
    phi_dot_dot=-(pendulum.k)*pendulum.phi-pendulum.c*pendulum.phi_dot

    pre_phi=pendulum.phi-int.delta_t*pendulum.phi_dot+int.delta_t^2*phi_dot_dot/2

    pendulum.phi=(pendulum.phi*((2/(int.delta_t^2)-(pendulum.k)))+pre_phi*(pendulum.c/(2*int.delta_t)-1/int.delta_t^2))/(1/int.delta_t^2+pendulum.c/(2*int.delta_t))
    
    pendulum.phi_dot=  pendulum.phi_dot  + int.delta_t*phi_dot_dot

end
