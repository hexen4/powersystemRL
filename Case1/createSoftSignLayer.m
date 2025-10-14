function softsignLayer = createSoftSignLayer(name)
    if nargin < 1
        name = 'softsign';
    end
    
    softsignLayer = functionLayer(...
        @forwardSoftSign, ...
        @backwardSoftSign, ...  
        'Formattable', true, ...
        'Name', name);
end

function Y = forwardSoftSign(X)
    % Forward: f(x) = x / (1 + |x|)
    Y = X ./ (1 + abs(X));
end

function dLdX = backwardSoftSign(X, Y, dLdY, memory)
    % Backward: f'(x) = 1 / (1 + |x|)^2 
    dYdX = 1 ./ ((1 + abs(X)).^2);
    dLdX = dLdY .* dYdX;
end