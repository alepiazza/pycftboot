def scalar_blocks_read(block_table, name):
    """
    This reads in a block table produced by scalar_blocks, the program by Walter
    Landry. Whether to call it is determined by `ConformalBlockTable`
    automatically.
    """
    files1 = os.listdir(name)
    files0 = sorted(files1)
    files = sorted(files0, key = len)
    # A cheap way to get alphanumeric sort
    info = files[0]

    # The convolution functions to support both can be found in the git history
    if info[:13] == "zzbDerivTable":
        print("Please rerun scalar_blocks with --output-ab")
        return
    elif info[:12] != "abDerivTable":
        print("Unknown convention for derivatives")
        return

    # Parsing is annoying because '-' is used in the numbers and the delimiters
    delta12_negative = info.split("-delta12--")
    delta12_positive = info.split("-delta12-")
    if len(delta12_negative) > 1:
        block_table.delta_12 = float(delta12_negative[1].split('-')[0])
        info = info.replace("-delta12--", "-delta12-")
    else:
        block_table.delta_12 = float(delta12_positive[1].split('-')[0])
    delta34_negative = info.split("-delta34--")
    delta34_positive = info.split("-delta34-")
    if len(delta34_negative) > 1:
        block_table.delta_34 = float(delta34_negative[1].split('-')[0])
        info = info.replace("-delta34--", "-delta34-")
    else:
        block_table.delta_34 = float(delta34_positive[1].split('-')[0])

    info = info.split('-')
    block_table.dim = float(info[1][1:])
    block_table.k_max = int(info[8][13:])
    block_table.n_max = int(info[7][4:]) - 1
    block_table.m_max = 1
    block_table.l_max = len(files) - 1
    block_table.odd_spins = False
    block_table.m_order = []
    block_table.n_order = []
    for n in range(0, block_table.n_max + 1):
        for m in range(0, 2 * (block_table.n_max - n) + 2):
            block_table.m_order.append(m)
            block_table.n_order.append(n)

    block_table.table = []
    for f in files:
        remove_zero = 0
        info = f.replace('--', '-')
        full = name + "/" + f
        l = int(info.split('-')[6][1:])
        if l % 2 == 1:
            block_table.odd_spins = True
        if l > block_table.l_max:
            block_table.l_max = l

        derivatives = []
        vector_with_poles = open(full, 'r').read().replace('{', '').replace('}', '').split("Poles -> ")
        if len(vector_with_poles) < 3:
            print("Please rerun scalar_blocks with --output-poles")
            return
        vector = vector_with_poles[0]
        single_poles = vector_with_poles[1].split(',\n')
        double_poles = vector_with_poles[2].split(',\n')
        vector = re.sub("abDeriv\[[0-9]+,[0-9]+\]", "", vector).split(',\n')[:-1]
        for el in vector:
            poly = 0
            poly_lines = el.split('\n')
            for k in range(0, len(poly_lines)):
                if k == 0:
                    coeff = poly_lines[k].split('->')[1]
                else:
                    coeff = poly_lines[k].split('*')[0][5:]
                poly += RealMPFR(coeff, prec) * (delta ** k)
            # It turns out that the scalars come with a shift of d - 2 which is not the unitarity bound
            # All shifts, scalar or not, are undone here as we prefer to handle this step during XML writing
            derivatives.append(poly.subs(delta, delta - block_table.dim - l + 2).expand())

        poles = []
        for p in range(0, len(single_poles)):
            pole = single_poles[p]
            if p != 0:
                pole = pole[17:]
            if pole == '' or pole == '\n':
                continue
            pole = RealMPFR(pole, prec)
            if l == 0 and abs(pole) < tiny:
                remove_zero = 1
                continue
            poles.append(pole)
        for p in range(0, len(double_poles)):
            pole = double_poles[p]
            if p != 0:
                pole = pole[17:]
            if pole == '' or pole == '\n':
                continue
            pole = RealMPFR(pole, prec)
            if l == 0 and abs(pole) < tiny:
                remove_zero = 2
                continue
            poles.append(pole)
            poles.append(pole)

        # The block for scalar exchange should not give zero for the identity
        if remove_zero > 0:
            for i in range(0, len(derivatives)):
                poly = 0
                coeffs = coefficients(derivatives[i])
                for c in range(remove_zero, len(coeffs)):
                    poly += coeffs[c] * (delta ** (c - remove_zero))
                derivatives[i] = poly
        block_table.table.append(PolynomialVector(derivatives, [l, 0], poles))

def scalar_blocks_write(block_table, name):
    """
    This writes out a block table in the format that scalar_blocks uses. It is
    triggered when a `ConformalBlockTable` is dumped with the right format string.
    """
    os.makedirs(name)
    name_prefix = "abDerivTable-d" + str(block_table.dim) + "-delta12-" + str(block_table.delta_12) + "-delta34-" + str(block_table.delta_34) + "-L"
    name_suffix = "-nmax" + str(block_table.n_max + 1) + "-keptPoleOrder" + str(block_table.k_max) + "-order" + str(block_table.k_max) + ".m"
    for l in range(0, len(block_table.table)):
        full = name + "/" + name_prefix + str(block_table.table[l].label[0]) + name_suffix
        block_file = open(full, 'w')
        block_file.write('{')
        for i in range(0, len(block_table.table[l].vector)):
            poly = block_table.table[l].vector[i].subs(delta, delta + block_table.dim + l - 2).expand()
            coeffs = coefficients(poly)
            block_file.write("abDeriv[" + str(block_table.m_order[i]) + "," + str(block_table.n_order[i]) + "] -> " + str(coeffs[0]) + "\n  ")
            for c in range(1, len(coeffs)):
                block_file.write(" + " + str(coeffs[c]) + "*x^" + str(c))
                if c == len(coeffs) - 1:
                    block_file.write(",\n ")
                else:
                    block_file.write("\n  ")

        single_poles = []
        double_poles = []
        gathered_poles = gather(block_table.table[l].poles)
        for p in gathered_poles.keys():
            if gathered_poles[p] == 1:
                single_poles.append(p)
            else:
                double_poles.append(p)

        block_file.write("singlePoles -> {")
        for p in range(0, len(single_poles)):
            block_file.write(str(single_poles[p]))
            if p < len(single_poles) - 1:
                block_file.write(",\n                 ")
        block_file.write("},\n doublePoles -> {")
        for p in range(0, len(double_poles)):
            block_file.write(str(double_poles[p]))
            if p < len(double_poles) - 1:
                block_file.write(",\n                 ")
        block_file.write('}}')
        block_file.close()
