function s = sigmoid( z )
	s = zeros(size(z));
    s = 1 ./ ( 1 + exp( -1 * z ));
end