import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

# common camera matrix for all scenarios



def get_2d_projection_of_3d_point(camera_matrix,point3d):
    '''

    :param camera_matrix:
    :param point3d:
    :return:
    '''
    projection = camera_matrix.dot(np.append(point3d,1))
    point_2d = projection/projection[2]
    return point_2d[0:2]

def get_log_likelihood(r_s,q_i,q_f,t_ratios,sigma_2d):
    '''

    :param r_s:
    :param q_i:
    :param q_f:
    :param t_ratios:
    :param sigma_2d:
    :param r_prime:
    :return:
    '''
    num_points = r_s.shape[0]
    q_s = np.zeros(shape=(num_points, 2))
    for i in range(0, num_points):
        q_s[i] = q_i + (q_f - q_i) * t_ratios[i]

    log_likelihood = np.zeros(shape=(num_points, 2))
    for i in range(0, num_points):
        log_likelihood[i] = multivariate_normal.logpdf(r_s[i], q_s[i], sigma_2d)


    return log_likelihood

def get_prior(p_i,p_f,mu_3d,sigma_3d):
    '''

    :param p_i:
    :param p_f:
    :param mu_3d:
    :param sigma_3d:
    :return:
    '''
    return multivariate_normal.logpdf([p_i,p_f],mu_3d,sigma_3d)

def proposal(previous,sigma):
    '''

    :param previous:
    :param sigma:
    :return:
    '''
    return multivariate_normal.rvs(previous,sigma)

def plot_parameter(x,y,x_label,y_label,title):
    '''

    :param x:
    :param y:
    :param x_label:
    :param y_label:
    :param title:
    :return:
    '''
    figure = plt.figure()

    figure.suptitle(title)
    plt.plot(x,y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()



def mh_sample_posterior_3d(camera_matrix,t_ratios,r_points,sigma_2d,mu_3d,sigma_3d,p_i=None,p_f=None,r_prime=None,new_camera_matrix=None):
   '''

   :param camera_matrix: matrix M
   :param t_ratios: interpolation ratios for the 2D points
   :param r_points: points from points_2d_camera_1.csv
   :param sigma_2d: Covariance for the likelihood distribution
   :param mu_3d: Mean the prior distribution
   :param sigma_3d: Covariance for the prior distribution
   :param p_i: Current p_i, initially sampled from prior distribution
   :param p_f: Current p_f, initially sampled from prior distribution
   :param r_prime: points from points_2d_camera_2.csv, required for task 5 in computing likelihood
   :param new_camera_matrix: M_prime, for points in points_2d_camera_2.csv
   :return: new_pi, new_pf, posterior probability, True/False (Whether new point was Accepted or not)
   '''

   # project current p_i,p_f onto 2D plane
   q_i = get_2d_projection_of_3d_point(camera_matrix,p_i)
   q_f = get_2d_projection_of_3d_point(camera_matrix,p_f)

   # Get prior distribution
   log_prior = get_prior(p_i,p_f,mu_3d,sigma_3d)

   # Compute log likelihood for r_points
   log_likelihood = get_log_likelihood(r_points,q_i,q_f,t_ratios,sigma_2d)

   # If points_2d_camera_2.csv is provided like in task 5, then add likelihood due to those points
   if r_prime is not None:
       q_i_prime = get_2d_projection_of_3d_point(new_camera_matrix,p_i)
       q_f_prime = get_2d_projection_of_3d_point(new_camera_matrix,p_f)
       log_likelihood = log_likelihood + get_log_likelihood(r_prime,q_i_prime,q_f_prime,t_ratios,sigma_2d)


   log_posterior = np.sum(log_likelihood) + np.sum(log_prior)
   p_i_next = proposal(p_i,sigma_3d)
   p_f_next = proposal(p_f,sigma_3d)

   q_i_next = get_2d_projection_of_3d_point(camera_matrix,p_i_next)
   q_f_next = get_2d_projection_of_3d_point(camera_matrix,p_f_next)

   log_likelihood_next = get_log_likelihood(r_points,q_i_next,q_f_next,t_ratios,sigma_2d)

   if r_prime is not None:
       q_i_prime_next = get_2d_projection_of_3d_point(new_camera_matrix,p_i_next)
       q_f_prime_next = get_2d_projection_of_3d_point(new_camera_matrix,p_f_next)
       log_likelihood_next = log_likelihood_next + get_log_likelihood(r_prime,q_i_prime_next,q_f_prime_next,t_ratios,sigma_2d)

   log_prior_next = get_prior(p_i_next,p_f_next,mu_3d,sigma_3d)

   log_posterior_next = np.sum(log_likelihood_next) + np.sum(log_prior_next)

   log_r = log_posterior_next - log_posterior
   if log_r >=0:
       return p_i_next,p_f_next,log_posterior_next,True
   else:
       u = np.random.uniform()
       if log_r >= np.log(u):
           return p_i_next,p_f_next,log_posterior_next,True
       else:
           return p_i,p_f,log_posterior,False





