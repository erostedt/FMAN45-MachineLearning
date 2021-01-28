function dldx = relu_backward(x, dldy)

    % Added code.
    sympref('HeavisideAtOrigin', 0);
    dldx = dldy.*heaviside(x);

end
