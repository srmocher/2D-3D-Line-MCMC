from mhsampling import mh_sample_posterior_3d,plot_parameter,get_2d_projection_of_3d_point
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

camera_matrix = np.array([[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 1, 0]])

new_camera_matrix = np.array([[0, 0, 1, -5], [0, 1, 0, 0], [-1, 0, 0, 5]])
num_iterations = 50000


def task2(r_points,t_ratios,r_prime,new_camera_matrix=None):
    '''

    :param r_points:
    :param t_ratios:
    :param r_prime:
    :param new_camera_matrix:
    :return:
    '''
    sigma_3d = 6 * np.identity(3)
    mu_3d = np.array([0, 0, 4]).T
    sigma_2d = (0.05) * (0.05) * np.identity(2)
    pi_points = []
    pf_points = []

    p_i_current, p_f_current = multivariate_normal.rvs(mu_3d, sigma_3d, size=2)
    posterior_probs = np.empty(shape=(num_iterations,1))
    for i in range(0, num_iterations):
        if i%100 == 0:
            print(str(i)+" iterations complete")
        new_pi, new_pf, posterior, accepted = mh_sample_posterior_3d(camera_matrix,t_ratios, r_points, sigma_2d, mu_3d, sigma_3d,
                                                                     p_i_current, p_f_current,r_prime,new_camera_matrix)
        posterior_probs[i]=posterior
        pi_points.append(new_pi)
        pf_points.append(new_pf)
        if accepted:
            p_i_current = new_pi
            p_f_current = new_pf

    indices = np.linspace(0, num_iterations, num_iterations)
    p_i_x = np.array(pi_points)[:, 0]
    p_i_y = np.array(pi_points)[:, 1]
    p_i_z = np.array(pi_points)[:, 2]

    p_f_x = np.array(pf_points)[:, 0]
    p_f_y = np.array(pf_points)[:, 1]
    p_f_z = np.array(pf_points)[:, 2]
    plot_parameter(indices, p_i_x, "No of samples", r"$P_i$ x-coordinate", r"Variation of parameter $P_i(x)$")
    plot_parameter(indices, p_i_y, "No of samples", r"$P_i$ y-coordinate", r"Variation of parameter $P_i(y)$")
    plot_parameter(indices, p_i_z, "No of samples", r"$P_i$ z-coordinate", r"Variation of parameter $P_i(z)$")

    plot_parameter(indices, p_f_x, "No of samples", r"$P_f$ x-coordinate", r"Variation of parameter $P_f(x)$")
    plot_parameter(indices, p_f_y, "No of samples", r"$P_f$ y-coordinate", r"Variation of parameter $P_f(y)$")
    plot_parameter(indices, p_f_z, "No of samples", r"$P_f$ z-coordinate", r"Variation of parameter $P_f(z)$")

    return pi_points,pf_points,posterior_probs

def task3(r_points,pi_points,pf_points,posterior_probs):
    '''

    :param r_points:
    :param pi_points:
    :param pf_points:
    :param posterior_probs:
    :return:
    '''
    max_index_posterior = np.argmax(posterior_probs)
    p_i_map_estimate = pi_points[max_index_posterior]
    p_f_map_estimate = pf_points[max_index_posterior]
    print(r'MAP estimate obtained in task3 for $p_i$ is '+str(p_i_map_estimate))
    print(r'MAP estimate obtained in task3 for $p_i$ is '+str(p_f_map_estimate))


    q_i_map_estimate = get_2d_projection_of_3d_point(camera_matrix,p_i_map_estimate)
    q_f_map_estimate = get_2d_projection_of_3d_point(camera_matrix,p_f_map_estimate)
    r_x = r_points[:, 0]
    r_y = r_points[:, 1]
    figure = plt.figure()
    figure.suptitle('2-D projection of MAP estimate of 3D line along with points from points_2d_camera_1.csv')
    plt.scatter(r_x, r_y)
    q_x = np.array([q_i_map_estimate[0], q_f_map_estimate[0]])
    q_y = np.array([q_i_map_estimate[1], q_f_map_estimate[1]])


    plt.plot(q_x, q_y)
    plt.show()
    return max_index_posterior

def task4(max_index_posterior,pi_points,pf_points,r_prime):
    '''

    :param max_index_posterior:
    :param pi_points:
    :param pf_points:
    :param r_prime:
    :return:
    '''
    p_i_map_estimate = pi_points[max_index_posterior]
    p_f_map_estimate = pf_points[max_index_posterior]
    q_i_map_estimate = get_2d_projection_of_3d_point(new_camera_matrix,p_i_map_estimate)
    q_f_map_estimate = get_2d_projection_of_3d_point(new_camera_matrix,p_f_map_estimate)

    q_x = np.array([q_i_map_estimate[0], q_f_map_estimate[0]])
    q_y = np.array([q_i_map_estimate[1], q_f_map_estimate[1]])

    r_prime_x = r_prime[:,0]
    r_prime_y = r_prime[:,1]


    figure = plt.figure()
    figure.suptitle('2-D projection of MAP estimate from task 3 with camera 2 along with points from points_2d_camera_1.csv')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.scatter(r_prime_x,r_prime_y)

    plt.plot(q_x,q_y)
    plt.show()


def task5(r_points,t_ratios,r_prime):
    '''

    :param r_points:
    :param t_ratios:
    :param r_prime:
    :return:
    '''
    pi,pf,posterior_probs = task2(r_points,t_ratios,r_prime,new_camera_matrix)
    max_index_posterior = np.argmax(posterior_probs)
    p_i_map_estimate = pi[max_index_posterior]
    p_f_map_estimate = pf[max_index_posterior]
    print(r"MAP estimate for $p_i$ in task5 is "+str(p_i_map_estimate))
    print(r"MAP estimate for $p_f$ in task5 is "+str(p_f_map_estimate))


    q_i_map = get_2d_projection_of_3d_point(camera_matrix,p_i_map_estimate)
    q_f_map = get_2d_projection_of_3d_point(camera_matrix,p_f_map_estimate)

    q_i_map_prime = get_2d_projection_of_3d_point(new_camera_matrix,p_i_map_estimate)
    q_f_map_prime = get_2d_projection_of_3d_point(new_camera_matrix,p_f_map_estimate)

    q_x = np.array([q_i_map[0],q_f_map[0]])
    q_y = np.array([q_i_map[1],q_f_map[1]])

    fig1 = plt.figure()
    fig1.suptitle('2-D projection of MAP estimate using camera 1 with likelihood combining both camera sources')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(q_x,q_y)
    plt.scatter(r_points[:,0],r_points[:,1])

    fig2 = plt.figure()
    plt.xlabel('x')
    plt.ylabel('y')
    fig2.suptitle('2-D projection of MAP estimate using camera 2 with likelihood combining both camera sources')
    q_prime_x = np.array([q_i_map_prime[0],q_f_map_prime[0]])
    q_prime_y = np.array([q_i_map_prime[1],q_f_map_prime[1]])

    plt.plot(q_prime_x,q_prime_y)
    plt.scatter(r_prime[:,0],r_prime[:,1])
    plt.show()








# Plot parameters
r_points = np.genfromtxt('data/points_2d_camera_1.csv', delimiter=',')
r_prime = np.genfromtxt('data/points_2d_camera_2.csv',delimiter=',')
t_ratios = np.genfromtxt('data/inputs.csv', delimiter=',')
pi_points,pf_points,posterior_probs = task2(r_points,t_ratios,None)
max_index_post = task3(r_points,pi_points,pf_points,posterior_probs)
task4(max_index_post,pi_points,pf_points,r_prime)
task5(r_points,t_ratios,r_prime)