import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def amdahl_scaling( T, t_p, n_p ):
    assert n_p.all() > 0 and t_p <= T
    return T / ( ( T - t_p ) + ( t_p / n_p ) )

def amdahl_limit( T, t_p, n_p ):
    assert n_p.all() > 0 and t_p <= T
    return np.ones_like( n_p ) * ( T / ( T - t_p ) )

def plot_amdahl_scaling( min_n_p, max_n_p ):
    num_processors = np.linspace( min_n_p, max_n_p )
    T = 1.0
    ideal = amdahl_scaling( T, 1.0 * T, num_processors )

    fig, ax = plt.subplots(figsize=(9,6), dpi=120)
    ideal = ax.plot( num_processors, amdahl_scaling( T, 1.0 * T, num_processors ), '--', color='k', label='ideal' )
    ax.plot( num_processors, amdahl_scaling( T, 0.9 * T, num_processors ), '-^', color='g', label='t_p = 0.90 * T' )
    ax.plot( num_processors, amdahl_limit( T, 0.90 * T, num_processors ), ':', color='g', label='t_p = 0.90 * T limit')
    ax.plot( num_processors, amdahl_scaling( T, 0.95 * T, num_processors ), '-x', color='b', label = 't_p = 0.95 * T' )
    ax.plot( num_processors, amdahl_limit( T, 0.95 * T, num_processors ), ':', color='b', label='t_p = 0.99 * T limit')
    ax.plot( num_processors, amdahl_scaling( T, 0.99 * T, num_processors ), '-o', color='r', label = 't_p = 0.99 * T' )
    ax.plot( num_processors, amdahl_limit( T, 0.99 * T, num_processors ), ':', color='r', label='t_p = 0.99 * T limit')
    ax.legend()
    ax.set_ylabel( 'speedup' )
    ax.set_xlabel( 'num processors n_p' )
    plt.xlim( min_n_p, max_n_p )
    plt.ylim( min_n_p, max_n_p )
    plt.title( "Amdahl's Law: Strong Scaling")
    plt.show()


def gustafson_scaling( T, t_p, n_p ):
    assert n_p.all() > 0 and t_p <= T
    f_p = t_p / T
    f_s = 1.0 - f_p
    scaling = f_s + f_p * n_p
    return scaling

def plot_gustafson_scaling( min_n_p, max_n_p ):
    num_processors = np.linspace( min_n_p, max_n_p )
    T = 1.0
    ideal = amdahl_scaling( T, 1.0 * T, num_processors )

    fig, ax = plt.subplots(figsize=(9,6), dpi=120)
    ax.plot( num_processors, gustafson_scaling( T, 1.00 * T, num_processors ), '--', color='k', label = 'ideal' )
    ax.plot( num_processors, gustafson_scaling( T, 0.90 * T, num_processors ), '-^', color='g', label = 't_p = 0.90 * T' )
    ax.plot( num_processors, gustafson_scaling( T, 0.95 * T, num_processors ), '-x', color='b', label = 't_p = 0.95 * T' )
    ax.plot( num_processors, gustafson_scaling( T, 0.99 * T, num_processors ), '-o', color='r', label = 't_p = 0.99 * T' )
    ax.legend()
    ax.set_ylabel( 'Scaled Speedup' )
    ax.set_xlabel( 'num processors n_p' )
    plt.xlim( min_n_p, max_n_p )
    plt.ylim( min_n_p, max_n_p )
    plt.title( "Gustafson-Barsis Law: Scaled Speedup")
    plt.show()
