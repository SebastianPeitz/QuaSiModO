#! /usr/bin/env python3
#
def hypercube_grid_points(m, n, ns, a, b, c):
    # *****************************************************************************80
    #
    ## HYPERCUBE_GRID_POINTS: grid points in a hypercube in M dimensions.
    #
    #  Discussion:
    #
    #    In M dimensional space, a logically rectangular grid is to be created.
    #    In the I-th dimension, the grid will use S(I) points.
    #    The total number of grid points is
    #      N = product ( 1 <= I <= M ) S(I)
    #
    #    Over the interval [A(i),B(i)], we have 5 choices for grid centering:
    #      1: 0,   1/3, 2/3, 1
    #      2: 1/5, 2/5, 3/5, 4/5
    #      3: 0,   1/4, 2/4, 3/4
    #      4: 1/4, 2/4, 3/4, 1
    #      5: 1/8, 3/8, 5/8, 7/8
    #
    #  Licensing:
    #
    #    This code is distributed under the GNU LGPL license.
    #
    #  Modified:
    #
    #    20 April 2015
    #
    #  Author:
    #
    #    John Burkardt
    #
    #  Parameters:
    #
    #    Input, integer M, the spatial dimension.
    #
    #    Input, integer N, the number of points.
    #    N = product ( 1 <= I <= M ) NS(I).
    #
    #    Input, integer NS(M), the number of points along
    #    each dimension.
    #
    #    Input, real A(M), B(M), the endpoints for each dimension.
    #
    #    Input, integer ( kind = 4 ) C(M), the grid centering for each dimension.
    #    1 <= C(*) <= 5.
    #
    #    Output, real X(M,N) = X(M*S(1),S(2),...,S(M)), the points.
    #
    import numpy as np
    #
    #  Create the 1D grids in each dimension.
    #
    x = np.zeros((m, n))

    contig = 0
    rep = 0
    skip = 0

    for i in range(0, m):

        s = ns[i]

        xs = np.zeros(s)

        for j in range(0, s):

            if (c[i] == 1):

                if (s == 1):
                    xs[j] = 0.5 * (a[i] + b[i])
                else:
                    xs[j] = ((s - j) * a[i] \
                             + (j - 1) * b[i]) \
                            / (s - 1)
            elif (c[i] == 2):
                xs[j] = ((s - j + 1) * a[i] \
                         + (j) * b[i]) \
                        / (s + 1)
            elif (c[i] == 3):
                xs[j] = ((s - j + 1) * a[i] \
                         + (j - 1) * b[i]) \
                        / (s)
            elif (c[i] == 4):
                xs[j] = ((s - j) * a[i] \
                         + (j) * b[i]) \
                        / (s)
            elif (c[i] == 5):
                xs[j] = ((2 * s - 2 * j + 1) * a[i] \
                         + (2 * j - 1) * b[i]) \
                        / (2 * s)

        x, contig, rep, skip = \
            r8vec_direct_product(i, s, xs, m, n, x, contig, rep, skip)

    return x


def hypercube_grid_points_test01():
    # *****************************************************************************80
    #
    ## HYPERCUBE_GRID_TEST01 tests HYPERCUBE_GRID on a two dimensional example.
    #
    #  Licensing:
    #
    #    This code is distributed under the GNU LGPL license.
    #
    #  Modified:
    #
    #    20 April 2015
    #
    #  Author:
    #
    #    John Burkardt
    #
    import numpy as np
    import platform

    m = 2

    a = np.array([0.0, 0.0])
    b = np.array([1.0, 10.0])
    c = np.array([2, 4])
    ns = np.array([4, 5])

    n = np.prod(ns)

    print('')
    print('HYPERCUBE_GRID_POINTS_TEST01')
    print('  Python version: %s' % (platform.python_version()))
    print('  HYPERCUBE_GRID_POINTS creates a grid of points in a hypercube.')
    print('  Spatial dimension M = %d' % (m))
    print('  Number of grid points N = %d' % (n))
    print('')
    print('     I    NS     C      A         B')
    print('')
    for i in range(0, m):
        print('  %4d  %4d  %4d  %8.4f  %8.4f' % (i, ns[i], c[i], a[i], b[i]))

    x = hypercube_grid_points(m, n, ns, a, b, c)
    #
    #  Transpose the data.
    #
    x = np.transpose(x)
    #
    #  Print the points.
    #
    r8mat_print(n, m, x, '  Grid points:')
    #
    #  Write the data to a file.
    #
    filename = 'hypercube_grid_points_test01.xyz'
    r8mat_write(filename, n, m, x)
    print('')
    print('  Data written to the file "%s".' % (filename))
    #
    #  Terminate.
    #
    print('')
    print('HYPERCUBE_GRID_POINTS_TEST01')
    print('  Normal end of execution.')
    return


def hypercube_grid_points_test02():
    # *****************************************************************************80
    #
    ## HYPERCUBE_GRID_TEST02 tests HYPERCUBE_GRID on a five dimensional example.
    #
    #  Licensing:
    #
    #    This code is distributed under the GNU LGPL license.
    #
    #  Modified:
    #
    #    20 April 2015
    #
    #  Author:
    #
    #    John Burkardt
    #
    import numpy as np
    import platform

    m = 5

    a = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    b = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    c = np.array([1, 2, 3, 4, 5])
    ns = np.array([2, 2, 2, 2, 2])

    n = np.prod(ns)

    print('')
    print('HYPERCUBE_GRID_POINTS_TEST02')
    print('  Python version: %s' % (platform.python_version()))
    print('  HYPERCUBE_GRID_POINTS creates a grid of points in a hypercube.')
    print('  Spatial dimension M = %d' % (m))
    print('  Number of grid points N = %d' % (n))
    print('')
    print('     I    NS     C      A         B')
    print('')
    for i in range(0, m):
        print('  %4d  %4d  %4d  %8.4f  %8.4f' % (i, ns[i], c[i], a[i], b[i]))

    x = hypercube_grid_points(m, n, ns, a, b, c)
    #
    #  Transpose the data.
    #
    x = np.transpose(x)
    #
    #  Print the points.
    #
    r8mat_print(n, m, x, '  Grid points:')
    #
    #  Write the data to a file.
    #
    filename = 'hypercube_grid_points_test02.xyz'
    r8mat_write(filename, n, m, x)
    print('')
    print('  Data written to the file "%s".' % (filename))
    #
    #  Terminate.
    #
    print('')
    print('HYPERCUBE_GRID_POINTS_TEST02')
    print('  Normal end of execution.')
    return


def hypercube_grid_points_test03():
    # *****************************************************************************80
    #
    ## HYPERCUBE_GRID_TEST03 tests HYPERCUBE_GRID on a three dimensional example.
    #
    #  Licensing:
    #
    #    This code is distributed under the GNU LGPL license.
    #
    #  Modified:
    #
    #    20 April 2015
    #
    #  Author:
    #
    #    John Burkardt
    #
    import numpy as np
    import platform

    m = 3

    a = np.array([-1.0, -1.0, -1.0])
    b = np.array([+1.0, +1.0, +1.0])
    c = np.array([1, 1, 1])
    ns = np.array([3, 3, 3])

    n = np.prod(ns)

    print('')
    print('HYPERCUBE_GRID_POINTS_TEST03')
    print('  Python version: %s' % (platform.python_version()))
    print('  HYPERCUBE_GRID_POINTS creates a grid of points in a hypercube.')
    print('  Spatial dimension M = %d' % (m))
    print('  Number of grid points N = %d' % (n))
    print('')
    print('     I    NS     C      A         B')
    print('')
    for i in range(0, m):
        print('  %4d  %4d  %4d  %8.4f  %8.4f' % (i, ns[i], c[i], a[i], b[i]))

    x = hypercube_grid_points(m, n, ns, a, b, c)
    #
    #  Transpose the data.
    #
    x = np.transpose(x)
    #
    #  Print the points.
    #
    r8mat_print(n, m, x, '  Grid points:')
    #
    #  Write the data to a file.
    #
    filename = 'hypercube_grid_points_test03.xyz'
    r8mat_write(filename, n, m, x)
    print('')
    print('  Data written to the file "%s".' % (filename))
    #
    #  Terminate.
    #
    print('')
    print('HYPERCUBE_GRID_POINTS_TEST03')
    print('  Normal end of execution.')
    return


def hypercube_grid_test():
    # *****************************************************************************80
    #
    ## HYPERCUBE_GRID_TEST tests the HYPERCUBE_GRID library.
    #
    #  Licensing:
    #
    #    This code is distributed under the GNU LGPL license.
    #
    #  Modified:
    #
    #    20 April 2015
    #
    #  Author:
    #
    #    John Burkardt
    #
    print('')
    print('HYPERCUBE_GRID_TEST')
    print('  Python version:')
    print('  Test the HYPERCUBE_GRID library.')
    #
    #  Utilities:
    #
    r8mat_print_test()
    r8mat_print_some_test()
    r8mat_write_test()
    timestamp_test()
    #
    #  Library.
    #
    hypercube_grid_points_test01()
    hypercube_grid_points_test02()
    hypercube_grid_points_test03()
    #
    #  Terminate.
    #
    print('')
    print('HYPERCUBE_GRID_TEST:')
    print('  Normal end of execution.')


def r8mat_print(m, n, a, title):
    # *****************************************************************************80
    #
    ## R8MAT_PRINT prints an R8MAT.
    #
    #  Licensing:
    #
    #    This code is distributed under the GNU LGPL license.
    #
    #  Modified:
    #
    #    31 August 2014
    #
    #  Author:
    #
    #    John Burkardt
    #
    #  Parameters:
    #
    #    Input, integer M, the number of rows in A.
    #
    #    Input, integer N, the number of columns in A.
    #
    #    Input, real A(M,N), the matrix.
    #
    #    Input, string TITLE, a title.
    #
    r8mat_print_some(m, n, a, 0, 0, m - 1, n - 1, title)

    return


def r8mat_print_test():
    # *****************************************************************************80
    #
    ## R8MAT_PRINT_TEST tests R8MAT_PRINT.
    #
    #  Licensing:
    #
    #    This code is distributed under the GNU LGPL license.
    #
    #  Modified:
    #
    #    10 February 2015
    #
    #  Author:
    #
    #    John Burkardt
    #
    import numpy as np
    import platform

    print('')
    print('R8MAT_PRINT_TEST')
    print('  Python version: %s' % (platform.python_version()))
    print('  R8MAT_PRINT prints an R8MAT.')

    m = 4
    n = 6
    v = np.array([ \
        [11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
        [21.0, 22.0, 23.0, 24.0, 25.0, 26.0],
        [31.0, 32.0, 33.0, 34.0, 35.0, 36.0],
        [41.0, 42.0, 43.0, 44.0, 45.0, 46.0]], dtype=np.float64)
    r8mat_print(m, n, v, '  Here is an R8MAT:')
    #
    #  Terminate.
    #
    print('')
    print('R8MAT_PRINT_TEST:')
    print('  Normal end of execution.')
    return


def r8mat_print_some(m, n, a, ilo, jlo, ihi, jhi, title):
    # *****************************************************************************80
    #
    ## R8MAT_PRINT_SOME prints out a portion of an R8MAT.
    #
    #  Licensing:
    #
    #    This code is distributed under the GNU LGPL license.
    #
    #  Modified:
    #
    #    10 February 2015
    #
    #  Author:
    #
    #    John Burkardt
    #
    #  Parameters:
    #
    #    Input, integer M, N, the number of rows and columns of the matrix.
    #
    #    Input, real A(M,N), an M by N matrix to be printed.
    #
    #    Input, integer ILO, JLO, the first row and column to print.
    #
    #    Input, integer IHI, JHI, the last row and column to print.
    #
    #    Input, string TITLE, a title.
    #
    incx = 5

    print('')
    print(title)

    if (m <= 0 or n <= 0):
        print('')
        print('  (None)')
        return

    for j2lo in range(max(jlo, 0), min(jhi + 1, n), incx):

        j2hi = j2lo + incx - 1
        j2hi = min(j2hi, n)
        j2hi = min(j2hi, jhi)

        print('')
        print('  Col: ', end='')

        for j in range(j2lo, j2hi + 1):
            print('%7d       ' % (j), end='')

        print('')
        print('  Row')

        i2lo = max(ilo, 0)
        i2hi = min(ihi, m)

        for i in range(i2lo, i2hi + 1):

            print('%7d :' % (i), end='')

            for j in range(j2lo, j2hi + 1):
                print('%12g  ' % (a[i, j]), end='')

            print('')

    return


def r8mat_print_some_test():
    # *****************************************************************************80
    #
    ## R8MAT_PRINT_SOME_TEST tests R8MAT_PRINT_SOME.
    #
    #  Licensing:
    #
    #    This code is distributed under the GNU LGPL license.
    #
    #  Modified:
    #
    #    31 October 2014
    #
    #  Author:
    #
    #    John Burkardt
    #
    import numpy as np
    import platform

    print('')
    print('R8MAT_PRINT_SOME_TEST')
    print('  Python version: %s' % (platform.python_version()))
    print('  R8MAT_PRINT_SOME prints some of an R8MAT.')

    m = 4
    n = 6
    v = np.array([ \
        [11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
        [21.0, 22.0, 23.0, 24.0, 25.0, 26.0],
        [31.0, 32.0, 33.0, 34.0, 35.0, 36.0],
        [41.0, 42.0, 43.0, 44.0, 45.0, 46.0]], dtype=np.float64)
    r8mat_print_some(m, n, v, 0, 3, 2, 5, '  Here is an R8MAT:')
    #
    #  Terminate.
    #
    print('')
    print('R8MAT_PRINT_SOME_TEST:')
    print('  Normal end of execution.')
    return


def r8mat_transpose_print(m, n, a, title):
    # *****************************************************************************80
    #
    ## R8MAT_TRANSPOSE_PRINT prints an R8MAT, transposed.
    #
    #  Licensing:
    #
    #    This code is distributed under the GNU LGPL license.
    #
    #  Modified:
    #
    #    31 August 2014
    #
    #  Author:
    #
    #    John Burkardt
    #
    #  Parameters:
    #
    #    Input, integer M, the number of rows in A.
    #
    #    Input, integer N, the number of columns in A.
    #
    #    Input, real A(M,N), the matrix.
    #
    #    Input, string TITLE, a title.
    #
    r8mat_transpose_print_some(m, n, a, 0, 0, m - 1, n - 1, title)

    return


def r8mat_transpose_print_test():
    # *****************************************************************************80
    #
    ## R8MAT_TRANSPOSE_PRINT_TEST tests R8MAT_TRANSPOSE_PRINT.
    #
    #  Licensing:
    #
    #    This code is distributed under the GNU LGPL license.
    #
    #  Modified:
    #
    #    31 October 2014
    #
    #  Author:
    #
    #    John Burkardt
    #
    import numpy as np
    import platform

    print('')
    print('R8MAT_TRANSPOSE_PRINT_TEST')
    print('  Python version: %s' % (platform.python_version()))
    print('  R8MAT_TRANSPOSE_PRINT prints an R8MAT.')

    m = 4
    n = 3
    v = np.array([ \
        [11.0, 12.0, 13.0],
        [21.0, 22.0, 23.0],
        [31.0, 32.0, 33.0],
        [41.0, 42.0, 43.0]], dtype=np.float64)
    r8mat_transpose_print(m, n, v, '  Here is an R8MAT, transposed:')
    #
    #  Terminate.
    #
    print('')
    print('R8MAT_TRANSPOSE_PRINT_TEST:')
    print('  Normal end of execution.')
    return


def r8mat_transpose_print_some(m, n, a, ilo, jlo, ihi, jhi, title):
    # *****************************************************************************80
    #
    ## R8MAT_TRANSPOSE_PRINT_SOME prints a portion of an R8MAT, transposed.
    #
    #  Licensing:
    #
    #    This code is distributed under the GNU LGPL license.
    #
    #  Modified:
    #
    #    13 November 2014
    #
    #  Author:
    #
    #    John Burkardt
    #
    #  Parameters:
    #
    #    Input, integer M, N, the number of rows and columns of the matrix.
    #
    #    Input, real A(M,N), an M by N matrix to be printed.
    #
    #    Input, integer ILO, JLO, the first row and column to print.
    #
    #    Input, integer IHI, JHI, the last row and column to print.
    #
    #    Input, string TITLE, a title.
    #
    incx = 5

    print('')
    print(title)

    if (m <= 0 or n <= 0):
        print('')
        print('  (None)')
        return

    for i2lo in range(max(ilo, 0), min(ihi, m - 1), incx):

        i2hi = i2lo + incx - 1
        i2hi = min(i2hi, m - 1)
        i2hi = min(i2hi, ihi)

        print('')
        print('  Row: ', end='')

        for i in range(i2lo, i2hi + 1):
            print('%7d       ' % (i), end='')

        print('')
        print('  Col')

        j2lo = max(jlo, 0)
        j2hi = min(jhi, n - 1)

        for j in range(j2lo, j2hi + 1):

            print('%7d :' % (j), end='')

            for i in range(i2lo, i2hi + 1):
                print('%12g  ' % (a[i, j]), end='')

            print('')

    return


def r8mat_transpose_print_some_test():
    # *****************************************************************************80
    #
    ## R8MAT_TRANSPOSE_PRINT_SOME_TEST tests R8MAT_TRANSPOSE_PRINT_SOME.
    #
    #  Licensing:
    #
    #    This code is distributed under the GNU LGPL license.
    #
    #  Modified:
    #
    #    31 October 2014
    #
    #  Author:
    #
    #    John Burkardt
    #
    import numpy as np
    import platform

    print('')
    print('R8MAT_TRANSPOSE_PRINT_SOME_TEST')
    print('  Python version: %s' % (platform.python_version()))
    print('  R8MAT_TRANSPOSE_PRINT_SOME prints some of an R8MAT, transposed.')

    m = 4
    n = 6
    v = np.array([ \
        [11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
        [21.0, 22.0, 23.0, 24.0, 25.0, 26.0],
        [31.0, 32.0, 33.0, 34.0, 35.0, 36.0],
        [41.0, 42.0, 43.0, 44.0, 45.0, 46.0]], dtype=np.float64)
    r8mat_transpose_print_some(m, n, v, 0, 3, 2, 5, '  R8MAT, rows 0:2, cols 3:5:')
    #
    #  Terminate.
    #
    print('')
    print('R8MAT_TRANSPOSE_PRINT_SOME_TEST:')
    print('  Normal end of execution.')
    return


def r8mat_write(filename, m, n, a):
    # *****************************************************************************80
    #
    ## R8MAT_WRITE writes an R8MAT to a file.
    #
    #  Licensing:
    #
    #    This code is distributed under the GNU LGPL license.
    #
    #  Modified:
    #
    #    12 October 2014
    #
    #  Author:
    #
    #    John Burkardt
    #
    #  Parameters:
    #
    #    Input, string FILENAME, the name of the output file.
    #
    #    Input, integer M, the number of rows in A.
    #
    #    Input, integer N, the number of columns in A.
    #
    #    Input, real A(M,N), the matrix.
    #
    output = open(filename, 'w')

    for i in range(0, m):
        for j in range(0, n):
            s = '  %g' % (a[i, j])
            output.write(s)
        output.write('\n')

    output.close()

    return


def r8mat_write_test():
    # *****************************************************************************80
    #
    ## R8MAT_WRITE_TEST tests R8MAT_WRITE.
    #
    #  Licensing:
    #
    #    This code is distributed under the GNU LGPL license.
    #
    #  Modified:
    #
    #    12 October 2014
    #
    #  Author:
    #
    #    John Burkardt
    #
    import numpy as np
    import platform

    print('')
    print('R8MAT_WRITE_TEST:')
    print('  Python version: %s' % (platform.python_version()))
    print('  Test R8MAT_WRITE, which writes an R8MAT to a file.')

    filename = 'r8mat_write_test.txt'
    m = 5
    n = 3
    a = np.array(( \
        (1.1, 1.2, 1.3), \
        (2.1, 2.2, 2.3), \
        (3.1, 3.2, 3.3), \
        (4.1, 4.2, 4.3), \
        (5.1, 5.2, 5.3)))
    r8mat_write(filename, m, n, a)

    print('')
    print('  Created file "%s".' % (filename))
    #
    #  Terminate.
    #
    print('')
    print('R8MAT_WRITE_TEST:')
    print('  Normal end of execution.')
    return


def r8vec_direct_product(factor_index, factor_order, \
                         factor_value, factor_num, point_num, x, contig, rep, skip):
    # *****************************************************************************80
    #
    ## R8VEC_DIRECT_PRODUCT creates a direct product of R8VEC's.
    #
    #  Discussion:
    #
    #    To explain what is going on here, suppose we had to construct
    #    a multidimensional quadrature rule as the product of K rules
    #    for 1D quadrature.
    #
    #    The product rule will be represented as a list of points and weights.
    #
    #    The J-th item in the product rule will be associated with
    #      item J1 of 1D rule 1,
    #      item J2 of 1D rule 2,
    #      ...,
    #      item JK of 1D rule K.
    #
    #    In particular,
    #      X(J) = ( X(1,J1), X(2,J2), ..., X(K,JK))
    #    and
    #      W(J) = W(1,J1) * W(2,J2) * ... * W(K,JK)
    #
    #    So we can construct the quadrature rule if we can properly
    #    distribute the information in the 1D quadrature rules.
    #
    #    This routine carries out that task.
    #
    #    Another way to do this would be to compute, one by one, the
    #    set of all possible indices (J1,J2,...,JK), and then index
    #    the appropriate information.  An advantage of the method shown
    #    here is that you can process the K-th set of information and
    #    then discard it.
    #
    #  Example:
    #
    #    Rule 1:
    #      Order = 4
    #      X(1:4) = ( 1, 2, 3, 4 )
    #
    #    Rule 2:
    #      Order = 3
    #      X(1:3) = ( 10, 20, 30 )
    #
    #    Rule 3:
    #      Order = 2
    #      X(1:2) = ( 100, 200 )
    #
    #    Product Rule:
    #      Order = 24
    #      X(1:24) =
    #        ( 1, 10, 100 )
    #        ( 2, 10, 100 )
    #        ( 3, 10, 100 )
    #        ( 4, 10, 100 )
    #        ( 1, 20, 100 )
    #        ( 2, 20, 100 )
    #        ( 3, 20, 100 )
    #        ( 4, 20, 100 )
    #        ( 1, 30, 100 )
    #        ( 2, 30, 100 )
    #        ( 3, 30, 100 )
    #        ( 4, 30, 100 )
    #        ( 1, 10, 200 )
    #        ( 2, 10, 200 )
    #        ( 3, 10, 200 )
    #        ( 4, 10, 200 )
    #        ( 1, 20, 200 )
    #        ( 2, 20, 200 )
    #        ( 3, 20, 200 )
    #        ( 4, 20, 200 )
    #        ( 1, 30, 200 )
    #        ( 2, 30, 200 )
    #        ( 3, 30, 200 )
    #        ( 4, 30, 200 )
    #
    #  Licensing:
    #
    #    This code is distributed under the GNU LGPL license.
    #
    #  Modified:
    #
    #    10 April 2015
    #
    #  Author:
    #
    #    John Burkardt
    #
    #  Parameters:
    #
    #    Input, integer FACTOR_INDEX, the index of the factor being processed.
    #    The first factor processed must be factor 1#
    #
    #    Input, integer FACTOR_ORDER, the order of the factor.
    #
    #    Input, real FACTOR_VALUE(FACTOR_ORDER), the factor values
    #    for factor FACTOR_INDEX.
    #
    #    Input, integer FACTOR_NUM, the number of factors.
    #
    #    Input, integer POINT_NUM, the number of elements in the direct product.
    #
    #    Input/output, real X(FACTOR_NUM,POINT_NUM), the elements of the
    #    direct product.  On output, this has been updated by the latest factor.
    #
    #    Input/output, integer CONTIG, the number of consecutive values to set.
    #    The user should not set or alter this value.
    #
    #    Input/output, integer SKIP, the distance from the current value of START
    #    to the next location of a block of values to set.
    #    The user should not set or alter this value.
    #
    #    Input/output, integer REP, the number of blocks of values to set.
    #    The user should not set or alter this value.
    #
    #  Local Parameters:
    #
    #    Local, integer START, the first location of a block of values to set.
    #
    import numpy as np

    if (factor_index == 0):
        contig = 1
        skip = 1
        rep = point_num

    rep = (rep // factor_order)
    skip = skip * factor_order

    for j in range(0, factor_order):

        start = j * contig

        for k in range(0, rep):
            for l in range(start, start + contig):
                x[factor_index, l] = factor_value[j]
            start = start + skip

    contig = contig * factor_order

    return x, contig, rep, skip


def r8vec_direct_product_test():
    # *****************************************************************************80
    #
    ## R8VEC_DIRECT_PRODUCT_TEST tests R8VEC_DIRECT_PRODUCT.
    #
    #  Licensing:
    #
    #    This code is distributed under the GNU LGPL license.
    #
    #  Modified:
    #
    #    10 April 2015
    #
    #  Author:
    #
    #    John Burkardt
    #
    import numpy as np
    import platform

    factor_num = 3
    point_num = 24

    print('')
    print('R8VEC_DIRECT_PRODUCT_TEST')
    print('  Python version: %s' % (platform.python_version()))
    print('  R8VEC_DIRECT_PRODUCT forms the entries of a')
    print('  direct product of a given number of R8VEC factors.')

    x = np.zeros((factor_num, point_num))
    contig = 0
    skip = 0
    rep = 0

    for factor_index in range(0, factor_num):

        if (factor_index == 0):
            factor_order = 4
            factor_value = np.array([1.0, 2.0, 3.0, 4.0])
        elif (factor_index == 1):
            factor_order = 3
            factor_value = np.array([50.0, 60.0, 70.0])
        elif (factor_index == 2):
            factor_order = 2
            factor_value = np.array([800.0, 900.0])

        x, contig, rep, skip = r8vec_direct_product(factor_index, factor_order, \
                                                    factor_value, factor_num, point_num, x, contig, rep, skip);

    r8mat_transpose_print(factor_num, point_num, x, '  Matrix (transposed)')
    #
    #  Terminate.
    #
    print('')
    print('R8VEC_DIRECT_PRODUCT_TEST:')
    print('  Normal end of execution.')
    return


def timestamp():
    # *****************************************************************************80
    #
    ## TIMESTAMP prints the date as a timestamp.
    #
    #  Licensing:
    #
    #    This code is distributed under the GNU LGPL license.
    #
    #  Modified:
    #
    #    06 April 2013
    #
    #  Author:
    #
    #    John Burkardt
    #
    #  Parameters:
    #
    #    None
    #
    import time

    t = time.time()
    print(time.ctime(t))

    return None


def timestamp_test():
    # *****************************************************************************80
    #
    ## TIMESTAMP_TEST tests TIMESTAMP.
    #
    #  Licensing:
    #
    #    This code is distributed under the GNU LGPL license.
    #
    #  Modified:
    #
    #    03 December 2014
    #
    #  Author:
    #
    #    John Burkardt
    #
    #  Parameters:
    #
    #    None
    #
    import platform

    print('')
    print('TIMESTAMP_TEST:')
    print('  Python version: %s' % (platform.python_version()))
    print('  TIMESTAMP prints a timestamp of the current date and time.')
    print('')

    timestamp()
    #
    #  Terminate.
    #
    print('')
    print('TIMESTAMP_TEST:')
    print('  Normal end of execution.')
    return


if (__name__ == '__main__'):
    timestamp()
    hypercube_grid_test()
    timestamp()
