def logistic_tent_key(x, y, r, u, size):
    """
    x is the initial value of the logistic map,
    y is the initial value of the tent map,
    r is the parameter of the logistic map,
    u is the parameter of the tent map,
    size is the number of elements in the key.
    """

    key = []

    for i in range(size):   
        x = r*x*(1-x)   # The logistic equation
        y = u*y*min(y,1-y) # The tent equation
        key.append(int(((x+y)*pow(10, 16))%256))    # Converting the generated number to integer, between 0 to 255

    return key
