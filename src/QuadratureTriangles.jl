module QuadratureTriangles

__precompile__(true)

using StaticArrays

export ShapeFunction

import Base

Base.Val(I::Int) = Val{I}()

struct ShapeFunction{nnodel, nip, M, T}
    weights::SVector{nip,T}
    N::Vector{SMatrix{1, nnodel, T, nnodel}}
    dNds::Vector{SMatrix{2, nnodel, T, M}}
    dN3ds::SMatrix{2, 3, Float64, 6}
end

function ShapeFunction(nip::Int, nnodel::Int)
    Vnip = Val(nip)
    Vnnodel = Val(nnodel)
    N, dNds, dN3ds = inner_barrier(Vnip, Vnnodel)
    ShapeFunction(w_ip, N, dNds, dN3ds)
end

function ShapeFunction(nip::Val{T}, nnodel::Val{M}) where {T,M}
    x_ip, w_ip = ip_triangle(nip)
        # local coordinates and weights of points for integration of
        # velocity/pressure matrices
    N, dNds = shape_functions_triangles(x_ip, nnodel)
        # velocity shape functions and their derivatives
    dN3ds = @SMatrix [-1.0   1.0   0.0 # w.r.t. r
                    -1.0   0.0   1.0]
        # derivatives of linear (3-node) shape functions; used to calculate
        # each element's Jacobian
    ShapeFunction(w_ip, N, dNds, dN3ds)
end

function inner_barrier(Vnip, Vnnodel)
    x_ip, w_ip = ip_triangle(Vnip)
        # local coordinates and weights of points for integration of
        # velocity/pressure matrices
    N, dNds = shape_functions_triangles(x_ip, Vnnodel)
        # velocity shape functions and their derivatives
    dN3ds = @SMatrix [-1.0   1.0   0.0 # w.r.t. r
                    -1.0   0.0   1.0]
        # derivatives of linear (3-node) shape functions; used to calculate
        # each element's Jacobian  
    return N, dNds, dN3ds
end    

function ip_triangle(nip)
    # if nip === 1
    #     ipx,ipw = ip_triangle1()
    if nip === 3
        ipx,ipw = ip_triangle3()
    elseif nip === 6
        ipx,ipw = ip_triangle6()
    elseif nip === 7
        ipx,ipw = ip_triangle7()
    end
    return ipx,ipw
end

ip_triangle(::Val{3}) = ip_triangle3()
ip_triangle(::Val{6}) = ip_triangle6()
ip_triangle(::Val{7}) = ip_triangle7()

function ip_triangle3()

    # -- Allocations
    ipx = Array{Float64,2}(undef,3,2)
    ipw = Vector{Float64}(undef,3)
    # ipx = @MMatrix zeros(3,2)
    # ipw = @MVector zeros(3)

    # -- Integration point coordinates
    ipx[1,1] = 1/6;
    ipx[1,2] = 1/6;
    ipx[2,1] = 2/3;
    ipx[2,2] = 1/6;
    ipx[3,1] = 1/6;
    ipx[3,2] = 2/3;

    # -- Weights
    ipw[1] = 1/6;
    ipw[2] = 1/6;
    ipw[3] = 1/6;

    return SMatrix{3,2}(ipx),SVector{3}(ipw)

end

function ip_triangle6()

    # -- Allocations
    ipx = Array{Float64,2}(undef,6,2)
    ipw = Vector{Float64}(undef,6)

    # -- Integration point coordinates
    g1       = (8.0-sqrt(10.0) + sqrt(38.0-44.0*sqrt(2.0/5.0)))/18.0;
    g2       = (8.0-sqrt(10.0) - sqrt(38.0-44.0*sqrt(2.0/5.0)))/18.0;
    ipx[1,1] = 1.0-2.0*g1;              #0.108103018168070;
    ipx[1,2] = g1;                      #0.445948490915965;
    ipx[2,1] = g1;                      #0.445948490915965;
    ipx[2,2] = 1.0-2.0*g1;              #0.108103018168070;
    ipx[3,1] = g1;                      #0.445948490915965;
    ipx[3,2] = g1;                      #0.445948490915965;
    ipx[4,1] = 1.0-2.0*g2;              #0.816847572980459;
    ipx[4,2] = g2;                      #0.091576213509771;
    ipx[5,1] = g2;                      #0.091576213509771;
    ipx[5,2] = 1.0-2.0*g2;              #0.816847572980459;
    ipx[6,1] = g2;                      #0.091576213509771;
    ipx[6,2] = g2;                      #0.091576213509771;

    # -- Weights
    w1      = (620.0 + sqrt(213125.0-53320.0*sqrt(10.0)))/3720.0;
    w2      = (620.0 - sqrt(213125.0-53320.0*sqrt(10.0)))/3720.0;
    ipw[1]  =  w1;                       #0.223381589678011;
    ipw[2]  =  w1;                       #0.223381589678011;
    ipw[3]  =  w1;                       #0.223381589678011;
    ipw[4]  =  w2;                       #0.109951743655322;
    ipw[5]  =  w2;                       #0.109951743655322;
    ipw[6]  =  w2;                       #0.109951743655322;
    ipw     = 0.5.*ipw;

    return SMatrix{6,2}(ipx),SVector{6}(ipw)

end

function ip_triangle7()

    # -- Allocations
    ipx = Array{Float64,2}(undef,7,2)
    ipw = Vector{Float64}(undef,7)
    # ipx = @MMatrix zeros(7,2)
    # ipw = @MVector zeros(7)

    # -- Integration point coordinates
    g1       = (6.0 - sqrt(15.0))/21.0;
    g2       = (6.0 + sqrt(15.0))/21.0;
    ipx[1,1] = 1.0/3.0;                 #0.333333333333333;
    ipx[1,2] = 1.0/3.0;                 #0.333333333333333;
    ipx[2,1] = 1.0-2.0*g1;              #0.797426985353087;
    ipx[2,2] = g1;                      #0.101286507323456;
    ipx[3,1] = g1;                      #0.101286507323456;
    ipx[3,2] = 1.0-2.0*g1;              #0.797426985353087;
    ipx[4,1] = g1;                      #0.101286507323456;
    ipx[4,2] = g1;                      #0.101286507323456;
    ipx[5,1] = 1.0-2.0*g2;              #0.059715871789770;
    ipx[5,2] = g2;                      #0.470142064105115;
    ipx[6,1] = g2;                      #0.470142064105115;
    ipx[6,2] = 1.0-2.0*g2;              #0.059715871789770;
    ipx[7,1] = g2;                      #0.470142064105115;
    ipx[7,2] = g2;                      #0.470142064105115;

    # -- Weights
    w1      = (155.0 - sqrt(15.0))/1200.0;
    w2      = (155.0 + sqrt(15.0))/1200.0;
    ipw[1]  = 0.225;
    ipw[2]  = w1;                        #0.125939180544827;
    ipw[3]  = w1;                        #0.125939180544827;
    ipw[4]  = w1;                        #0.125939180544827;
    ipw[5]  = w2;                        #0.132394152788506;
    ipw[6]  = w2;                        #0.132394152788506;
    ipw[7]  = w2;                        #0.132394152788506;
    ipw     = 0.5.*ipw;

    return SMatrix{7,2}(ipx),SVector{7}(ipw)

end # END OF INTEGRATION POINT FUNCTIONS


# BEGIN OF INTEGRATION POINT SHAPE FUNCTIONS
function shape_functions_triangles(lc, nnodel)
    r    = lc[:,1]
    s    = lc[:,2]
    npt  = length(r)
    N, dN = get_N_∇N(r, s, npt, nnodel)
    return N, dN
end

function get_N_∇N(r,s,npt,::Val{3})
    N = [sf_N_tri3(r[ip],s[ip]) for ip in 1:npt]
    dN = [sf_dN_tri3(r[ip],s[ip]) for ip in 1:npt]
    return N, dN
end

function get_N_∇N(r,s,npt,::Val{6})
    N = [sf_N_tri6(r[ip],s[ip]) for ip in 1:npt]
    dN = [sf_dN_tri6(r[ip],s[ip]) for ip in 1:npt]
    return N, dN
end

function get_N_∇N(r,s,npt,::Val{7})
    N = [sf_N_tri7(r[ip],s[ip]) for ip in 1:npt]
    dN = [sf_dN_tri7(r[ip],s[ip]) for ip in 1:npt]
    return N, dN
end

function sf_dsf_tri7!(r,s,N,dN,ip)
    # Find shape functions and their derivatives at given points on the
    # master element for a 7 node triangle
    # 7-node triangle (node numbering is important)
    #
    #        3
    #        | \
    # s-axis 6   5
    #        | 7   \
    #        1 - 4 - 2
    #          r-axis
    t       = 1-r-s
    N[ip]  .= [t*(2*t-1)+3*r*s*t r*(2*r-1)+3*r*s*t s*(2*s-1)+3*r*s*t 4*r*t-12*r*s*t 4*r*s-12*r*s*t 4*s*t-12*r*s*t 27*r*s*t];
    dN[ip] .= [1-4*t+3*s*t-3*r*s  -1+4*r+3*s*t-3*r*s  3*s*t-3*r*s        4*t-4*r+12*r*s-12*s*t 4*s+12*r*s-12*s*t -4*s+12*r*s-12*s*t    -27*r*s+27*s*t
               1-4*t+3*r*t-3*r*s   3*r*t-3*r*s       -1+4*s+3*r*t-3*r*s -4*r-12*r*t+12*r*s     4*r-12*r*t+12*r*s  4*t-4*s-12*r*t+12*r*s 27*r*t-27*r*s]

    return N,dN
end # END OF FUNCTION sf_dsf_tri

function sf_dsf_tri6!(r,s,N,dN,ip)
    # Find shape functions and their derivatives at given points on the
    # master element for a 6 node triangle (node numbering is important)
    #
    #        3
    #        | \
    # s-axis 6   5
    #        |     \
    #        1 - 4 - 2
    #          r-axis
    #
    t = 1.0 - r - s
    # N1 at coordinate (r,s), N2 at coordinate (r,s), etc
    N[ip]  .= [t*(2.0*t-1.0)  r*(2.0*r-1.0) s*(2.0*s-1.0) 4.0*r*t 4.0*r*s 4.0*s*t];
    #     dN1       dN2    dN3    dN4       dN5    dN6
    dN[ip] .= [-(4.0*t-1.0)  4.0*r-1  0.0         4.0*(t-r)  4.0*s   -4.0*s     # w.r.t. r
               -(4.0*t-1.0)  0.0      4.0*s-1.0  -4.0*r      4.0*r   4.0*(t -s)]; # w.r.t. s

    return N,dN
end # END OF FUNCTION sf_dsf_tri

function sf_dsf_tri3!(r,s,N,dN,ip)
    # Find shape functions and their derivatives at given points on the
    # master element for 3 node triangle
    # 3-node triangle (node numbering is important)
    #
    #        3
    #        | \
    # s-axis |   \
    #        |     \
    #        1 - - - 2
    #          r axis -->
    t       = 1.0-r-s;
    N[ip]  .= [t r s];   # N3 at coordinate (r,s)
    dN[ip] .= [-1.0   1.0   0.0 # w.r.t. r
          -1.0   0.0   1.0]; # w.r.t. s
    return N,dN
end # END OF FUNCTION sf_dsf_tri
# END OF INTEGRATION POINT SHAPE FUNCTIONS

function sf_N_tri3(r,s)
    # Find shape functions and their derivatives at given points on the
    # master element for 3 node triangle
    # 3-node triangle (node numbering is important)
    #
    #        3
    #        | \
    # s-axis |   \
    #        |     \
    #        1 - - - 2
    #          r axis -->
    t       = 1.0-r-s;
    N = SMatrix{1,3}([t r s]);   # N3 at coordinate (r,s)
    return N
end # END OF FUNCTION sf_dsf_tri

function sf_dN_tri3(r,s)
    # Find shape functions and their derivatives at given points on the
    # master element for 3 node triangle
    # 3-node triangle (node numbering is important)
    #
    #        3
    #        | \
    # s-axis |   \
    #        |     \
    #        1 - - - 2
    #          r axis -->
    t       = 1.0-r-s;
    dN = @SMatrix [-1.0   1.0   0.0 # w.r.t. r
          -1.0   0.0   1.0]; # w.r.t. s
    return dN
end # END OF FUNCTION sf_dsf_tri

function sf_N_tri7(r,s)
    # Find shape functions and their derivatives at given points on the
    # master element for a 7 node triangle
    # 7-node triangle (node numbering is important)
    #
    #        3
    #        | \
    # s-axis 6   5
    #        | 7   \
    #        1 - 4 - 2
    #          r-axis
    t       = 1-r-s
    N = SMatrix{1,7}([t*(2*t-1)+3*r*s*t r*(2*r-1)+3*r*s*t s*(2*s-1)+3*r*s*t 4*r*t-12*r*s*t 4*r*s-12*r*s*t 4*s*t-12*r*s*t 27*r*s*t]);

    return N
end # END OF FUNCTION sf_dsf_tri

function sf_dN_tri7(r,s)
    # Find shape functions and their derivatives at given points on the
    # master element for a 7 node triangle
    # 7-node triangle (node numbering is important)
    #
    #        3
    #        | \
    # s-axis 6   5
    #        | 7   \
    #        1 - 4 - 2
    #          r-axis
    t       = 1-r-s
    dN= SMatrix{2,7}([1-4*t+3*s*t-3*r*s  -1+4*r+3*s*t-3*r*s  3*s*t-3*r*s        4*t-4*r+12*r*s-12*s*t 4*s+12*r*s-12*s*t -4*s+12*r*s-12*s*t    -27*r*s+27*s*t
               1-4*t+3*r*t-3*r*s   3*r*t-3*r*s       -1+4*s+3*r*t-3*r*s -4*r-12*r*t+12*r*s     4*r-12*r*t+12*r*s  4*t-4*s-12*r*t+12*r*s 27*r*t-27*r*s])

    return dN
end # END OF FUNCTION sf_dsf_tri

function sf_N_tri6(r,s)
    # Find shape functions and their derivatives at given points on the
    # master element for a 6 node triangle (node numbering is important)
    #
    #        3
    #        | \
    # s-axis 6   5
    #        |     \
    #        1 - 4 - 2
    #          r-axis
    #
    t = 1.0 - r - s
    # N1 at coordinate (r,s), N2 at coordinate (r,s), etc
    N = SMatrix{1,6}([t*(2.0*t-1.0)  r*(2.0*r-1.0) s*(2.0*s-1.0) 4.0*r*t 4.0*r*s 4.0*s*t]);
    #     dN1       dN2    dN3    dN4       dN5    dN6

    return N
end # END OF FUNCTION sf_dsf_tri

function sf_dN_tri6(r,s)
    # Find shape functions and their derivatives at given points on the
    # master element for a 6 node triangle (node numbering is important)
    #
    #        3
    #        | \
    # s-axis 6   5
    #        |     \
    #        1 - 4 - 2
    #          r-axis
    #
    t = 1.0 - r - s
    # N1 at coordinate (r,s), N2 at coordinate (r,s), etc
    #     dN1       dN2    dN3    dN4       dN5    dN6
    dN = SMatrix{2,6}([-(4.0*t-1.0)  4.0*r-1  0.0         4.0*(t-r)  4.0*s   -4.0*s     # w.r.t. r
               -(4.0*t-1.0)  0.0      4.0*s-1.0  -4.0*r      4.0*r   4.0*(t -s)]); # w.r.t. s

    return dN
end # END OF FUNCTION sf_dsf_tri

end
