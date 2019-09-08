@testset "gather" begin
    X = [1 2 3; 4 5 6]
    ind1 = [1 2 2; 1 1 1]
    ind2 = [1 2; 3 1]
    @test gather(X, ind1, 1) == [1 5 6; 1 2 3]
    @test gather(X, ind2, 2) == [1 2; 6 4]
end
