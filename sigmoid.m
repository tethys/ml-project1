function s = sigmoid( x )

    if x > 25
        s = exp( - log( 1 + exp( -x ) ) );
    else if x < -25
            s = exp( x - log( 1 + exp( x ) ) );
        else
                s = 1.0 ./ (1.0 + exp( -x ));
        end
    end

end