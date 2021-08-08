using  QuadratureTriangles

using BenchmarkTools

nip, nnodel = 3, 3 

@benchmark ShapeFunction($nip, $nnodel)